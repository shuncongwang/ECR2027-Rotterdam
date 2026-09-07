from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

from utils_dl import load_config, project_path, project_dir, set_seed, get_device
from nifti_dataset import NiftiClassificationDataset
from medicalnet_resnet import MedicalNetResNetClassifier, load_medicalnet_pretrained
from metrics_dl import choose_threshold_youden, calculate_binary_metrics


def predict(model, loader, device):
    model.eval()
    ys, scores, cases = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            ys.extend(batch["target"].cpu().numpy().tolist())
            scores.extend(prob.cpu().numpy().tolist())
            cases.extend(batch["case"])
    return np.asarray(ys), np.asarray(scores), cases


def make_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_resnet.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["training"]["random_state"])
    set_seed(seed)
    device = get_device(cfg["runtime"].get("device", "auto"))
    print(f"Device: {device}")

    #out_dir = project_path(cfg, "output_dir")
    out_dir = Path(__file__).resolve().parent / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "best_resnet_classifier.pt"
    history_file = out_dir / "training_history.xlsx"
    val_predictions_file = out_dir / "validation_predictions.xlsx"
    preload_report_file = out_dir / "pretrained_loading_report.json"

    df = pd.read_excel(project_path(cfg, "train_validation_xlsx_cleaned"))
    print(f"Training xlsx: {cfg['paths']['train_validation_xlsx_cleaned']}")
    case_col = cfg["columns"]["case"]
    target_col = cfg["columns"]["target"]
    df[case_col] = df[case_col].astype(str).str.strip()

    train_df, val_df = train_test_split(
        df,
        test_size=float(cfg["training"]["validation_fraction"]),
        random_state=seed,
        stratify=df[target_col].astype(int),
    )

    train_ds = NiftiClassificationDataset(train_df, cfg, training=True)
    val_ds = NiftiClassificationDataset(val_df, cfg, training=False)

    batch_size = int(cfg["training"]["batch_size"])
    workers = int(cfg["training"].get("num_workers", 0))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    in_channels = len(cfg["columns"]["image_columns"])
    model = MedicalNetResNetClassifier(
        depth=int(cfg["model"]["depth"]),
        in_channels=in_channels,
        num_classes=int(cfg["model"].get("num_classes", 2)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
    )

    use_pretrained = bool(cfg["model"].get("use_pretrained", False))
    ckpt_path = project_path(cfg, "pretrained_checkpoint")
    if use_pretrained:
        if ckpt_path is None:
            raise ValueError("model.use_pretrained=true but paths.pretrained_checkpoint is null")
        report = load_medicalnet_pretrained(
            model,
            ckpt_path,
            adapt_input_channels=bool(cfg["model"].get("adapt_pretrained_input_channels", True)),
        )
        preload_report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Loaded {report['loaded_count']} pretrained tensors; skipped {report['skipped_count']}.")
        if report["loaded_count"] == 0:
            raise RuntimeError("No pretrained tensors were loaded. Check depth/checkpoint compatibility.")

    model.to(device)

    freeze_epochs = int(cfg["model"].get("freeze_encoder_epochs", 0))
    if freeze_epochs > 0:
        model.set_encoder_trainable(False)
        print(f"Encoder frozen for first {freeze_epochs} epoch(s).")

    y_train = train_df[target_col].astype(int).to_numpy()
    if str(cfg["training"].get("class_weight", "balanced")).lower() == "balanced":
        weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    optimizer = make_optimizer(cfg["training"].get("optimizer", "adamw"), filter(lambda p: p.requires_grad, model.parameters()), lr, wd)

    scheduler_name = str(cfg["training"].get("scheduler", "cosine")).lower()
    epochs = int(cfg["training"]["epochs"])
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    else:
        scheduler = None

    amp_enabled = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    patience = int(cfg["training"].get("early_stopping_patience", 15))
    min_delta = float(cfg["training"].get("min_delta", 1e-4))
    best_auc = -np.inf
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.set_encoder_trainable(True)
            optimizer = make_optimizer(cfg["training"].get("optimizer", "adamw"), model.parameters(), lr, wd)
            if scheduler_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - freeze_epochs))
            elif scheduler_name == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
            print("Encoder unfrozen.")

        model.train()
        running_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / max(n_seen, 1)
        val_y, val_score, val_cases = predict(model, val_loader, device)
        val_auc = roc_auc_score(val_y, val_score)
        threshold_mode = str(cfg["training"].get("threshold_mode", "youden")).lower()
        if threshold_mode == "youden":
            threshold = choose_threshold_youden(val_y, val_score)
        else:
            threshold = float(cfg["training"].get("fixed_threshold", 0.5))
        val_metrics = calculate_binary_metrics(val_y, val_score, threshold)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc, "val_f1": val_metrics["F1"], "threshold": threshold, "lr": current_lr})
        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | val AUC={val_auc:.4f} | F1={val_metrics['F1']:.4f} | thr={threshold:.4f}")

        improved = val_auc > best_auc + min_delta
        if improved:
            best_auc = val_auc
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "depth": int(cfg["model"]["depth"]),
                "in_channels": in_channels,
                "num_classes": int(cfg["model"].get("num_classes", 2)),
                "dropout": float(cfg["model"].get("dropout", 0.2)),
                "validation_auc": float(val_auc),
                "threshold": float(threshold),
                "image_columns": list(cfg["columns"]["image_columns"]),
                "config": cfg,
            }, checkpoint_file)
            pd.DataFrame({case_col: val_cases, "True_Label": val_y, "Predicted_Score": val_score}).to_excel(val_predictions_file, index=False)
        else:
            stale += 1

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_auc)
            else:
                scheduler.step()

        pd.DataFrame(history).to_excel(history_file, index=False)

        if stale >= patience:
            print(f"Early stopping after {epoch} epochs. Best validation AUC={best_auc:.4f}")
            break

    print(f"Best checkpoint: {checkpoint_file}")


if __name__ == "__main__":
    main()

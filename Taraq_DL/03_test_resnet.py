from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix

from utils_dl import load_config, project_path, get_device, set_seed
from nifti_dataset import NiftiClassificationDataset
from medicalnet_resnet import MedicalNetResNetClassifier
from metrics_dl import calculate_binary_metrics, bootstrap_metrics


def predict(model, loader, device):
    model.eval()
    ys, scores, cases = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            ys.extend(batch["target"].numpy().tolist())
            scores.extend(prob.cpu().numpy().tolist())
            cases.extend(batch["case"])
    return np.asarray(ys), np.asarray(scores), cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_resnet.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["training"]["random_state"]))
    device = get_device(cfg["runtime"].get("device", "auto"))

    out_dir = project_path(cfg, "output_dir")
    ckpt_file = out_dir / "best_resnet_classifier.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Train first; checkpoint not found: {ckpt_file}")

    try:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_file, map_location="cpu")

    model = MedicalNetResNetClassifier(
        depth=int(ckpt["depth"]),
        in_channels=int(ckpt["in_channels"]),
        num_classes=int(ckpt.get("num_classes", 2)),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    df_test = pd.read_excel(project_path(cfg, "test_xlsx"))
    case_col = cfg["columns"]["case"]
    target_col = cfg["columns"]["target"]
    df_test[case_col] = df_test[case_col].astype(str).str.strip()

    ds = NiftiClassificationDataset(df_test, cfg, training=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    y, score, cases = predict(model, loader, device)
    threshold = float(ckpt["threshold"])  # selected only on validation data
    pred = (score >= threshold).astype(int)

    metrics = bootstrap_metrics(
        y, score, threshold,
        n_bootstrap=int(cfg["evaluation"].get("bootstrap_iterations", 2000)),
        ci=float(cfg["evaluation"].get("ci_level", 0.95)),
        seed=int(cfg["training"]["random_state"]),
    )

    pred_df = pd.DataFrame({
        case_col: cases,
        "True_Label": y,
        "Predicted_Label": pred,
        "Predicted_Score": score,
    })
    metric_df = pd.DataFrame([metrics])

    result_xlsx = out_dir / "resnet_test_performance.xlsx"
    with pd.ExcelWriter(result_xlsx, engine="openpyxl") as writer:
        metric_df.to_excel(writer, sheet_name="Performance_95CI", index=False)
        pred_df.to_excel(writer, sheet_name="Test_Predictions", index=False)

    if bool(cfg["evaluation"].get("save_roc_curve", True)):
        fpr, tpr, _ = roc_curve(y, score)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Independent Test ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "test_roc.png", dpi=300, bbox_inches="tight")
        plt.close()

    if bool(cfg["evaluation"].get("save_confusion_matrix", True)):
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        ax.set_title("Independent Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(out_dir / "test_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(metric_df[["AUC_95CI", "F1_95CI", "Sensitivity_95CI", "Specificity_95CI"]].to_string(index=False))
    print(f"Saved: {result_xlsx}")


if __name__ == "__main__":
    main()

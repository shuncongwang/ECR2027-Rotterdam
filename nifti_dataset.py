from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils_dl import resolve_data_path


def _read_nifti(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D,H,W]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {arr.shape}: {path}")
    return arr


def _bbox_from_mask(mask: np.ndarray, margin):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    margin = np.asarray(margin, dtype=int)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _resize_volume(arr: np.ndarray, output_size) -> np.ndarray:
    t = torch.from_numpy(arr)[None, None]  # [1,1,D,H,W]
    t = F.interpolate(t, size=tuple(output_size), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


def _resize_mask(arr: np.ndarray, output_size) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=tuple(output_size), mode="nearest")
    return t[0, 0].numpy() > 0.5


def _zscore(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = arr[valid]
    if vals.size < 2:
        vals = arr[np.isfinite(arr)]
    mean = float(vals.mean()) if vals.size else 0.0
    std = float(vals.std()) if vals.size else 1.0
    if std < 1e-6:
        std = 1.0
    return (arr - mean) / std


class NiftiClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, training: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.training = training
        self.root = Path(cfg["project"]["dir"])
        self.case_col = cfg["columns"]["case"]
        self.target_col = cfg["columns"]["target"]
        self.image_columns = list(cfg["columns"]["image_columns"])
        self.mask_column = cfg["columns"].get("mask_column")

        icfg = cfg["input"]
        self.output_size = tuple(int(x) for x in icfg["output_size"])
        self.normalization = icfg.get("normalization", "zscore_nonzero")
        self.apply_mask = bool(icfg.get("apply_mask", False))
        self.crop_to_mask = bool(icfg.get("crop_to_mask", False))
        self.crop_margin = tuple(int(x) for x in icfg.get("crop_margin_voxels", [0, 0, 0]))

        aug = icfg.get("augmentation", {})
        self.flip_p = float(aug.get("random_flip_probability", 0.0))
        self.noise_std = float(aug.get("gaussian_noise_std", 0.0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = str(row[self.case_col])
        target = int(row[self.target_col])

        mask = None
        if self.mask_column and self.mask_column in self.df.columns and pd.notna(row[self.mask_column]):
            mask_path = resolve_data_path(row[self.mask_column], self.root)
            mask = _read_nifti(mask_path) > 0

        volumes = []
        reference_shape = None

        for col in self.image_columns:
            if col not in self.df.columns:
                raise KeyError(f"Image column '{col}' not found in xlsx.")
            path = resolve_data_path(row[col], self.root)
            if not path.exists():
                raise FileNotFoundError(f"{case_id}: {path}")
            arr = _read_nifti(path)

            if reference_shape is None:
                reference_shape = arr.shape
            elif arr.shape != reference_shape:
                raise ValueError(
                    f"{case_id}: multi-sequence shapes differ {reference_shape} vs {arr.shape}. "
                    "Register/resample sequences to the same grid first."
                )

            volumes.append(arr)

        if mask is not None and reference_shape != mask.shape:
            raise ValueError(
                f"{case_id}: image/mask shapes differ {reference_shape} vs {mask.shape}. "
                "Use a mask on the same image grid."
            )

        # Crop every sequence identically using ROI bbox.
        if self.crop_to_mask and mask is not None:
            bbox = _bbox_from_mask(mask, self.crop_margin)
            if bbox is not None:
                volumes = [v[bbox] for v in volumes]
                mask = mask[bbox]

        # Resize after crop/full-volume selection.
        volumes = [_resize_volume(v, self.output_size) for v in volumes]
        if mask is not None:
            mask = _resize_mask(mask.astype(np.float32), self.output_size)

        normalized = []
        for arr in volumes:
            if self.normalization == "zscore_mask" and mask is not None and mask.any():
                valid = mask
            else:
                valid = np.isfinite(arr) & (arr != 0)
            arr = _zscore(arr, valid)
            if self.apply_mask and mask is not None:
                arr = arr * mask.astype(np.float32)
            normalized.append(arr.astype(np.float32))

        x = np.stack(normalized, axis=0)  # [C,D,H,W]

        if self.training:
            # spatial flips are applied identically across channels
            for axis in (1, 2, 3):
                if random.random() < self.flip_p:
                    x = np.flip(x, axis=axis).copy()
            if self.noise_std > 0:
                x = x + np.random.normal(0.0, self.noise_std, x.shape).astype(np.float32)

        return {
            "image": torch.from_numpy(x),
            "target": torch.tensor(target, dtype=torch.long),
            "case": case_id,
        }

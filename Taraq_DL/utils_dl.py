from __future__ import annotations

from pathlib import Path
from typing import Union
import random
import yaml
import numpy as np
import torch


def load_config(config_path: Union[str, Path]) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_dir(cfg: dict) -> Path:
    return Path(cfg["project"]["dir"])


def project_path(cfg: dict, key: str) -> Path:
    value = cfg["paths"][key]
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else project_dir(cfg) / p


def resolve_data_path(value, root: Path) -> Path:
    if value is None:
        raise ValueError("Path value is None")
    p = Path(str(value).strip())
    return p if p.is_absolute() else root / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    name = str(name).lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device='cuda' but CUDA is not available.")
    return torch.device(name)

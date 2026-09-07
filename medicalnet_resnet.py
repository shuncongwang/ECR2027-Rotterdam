"""Modernized MedicalNet-compatible 3D ResNet for classification.

The encoder follows the MedicalNet/Med3D ResNet naming convention closely enough
for encoder checkpoint transfer. The original segmentation head is intentionally
NOT reproduced. Classification uses:
    layer4 -> AdaptiveAvgPool3d(1) -> Dropout -> Linear
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_DEPTHS = {
    10: ("basic", [1, 1, 1, 1], "B"),
    18: ("basic", [2, 2, 2, 2], "A"),
    34: ("basic", [3, 4, 6, 3], "A"),
    50: ("bottleneck", [3, 4, 6, 3], "B"),
    101: ("bottleneck", [3, 4, 23, 3], "B"),
    152: ("bottleneck", [3, 8, 36, 3], "B"),
    200: ("bottleneck", [3, 24, 36, 3], "B"),
}


def _shortcut_a(x: torch.Tensor, out_channels: int, stride: int) -> torch.Tensor:
    # Modern, device-safe replacement for MedicalNet's old Variable/.cuda shortcut A.
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    ch = out.shape[1]
    if ch < out_channels:
        zeros = out.new_zeros(
            out.shape[0], out_channels - ch, out.shape[2], out.shape[3], out.shape[4]
        )
        out = torch.cat([out, zeros], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, shortcut_type: str = "B"):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        out_channels = planes * self.expansion
        self.shortcut_type = shortcut_type
        self.stride = stride
        self.out_channels = out_channels

        if stride != 1 or inplanes != out_channels:
            if shortcut_type == "B":
                self.downsample = nn.Sequential(
                    nn.Conv3d(inplanes, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_channels),
                )
            else:
                self.downsample = None
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is None:
            residual = _shortcut_a(x, self.out_channels, self.stride)
        else:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, shortcut_type: str = "B"):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        out_channels = planes * self.expansion
        self.shortcut_type = shortcut_type
        self.stride = stride
        self.out_channels = out_channels

        if stride != 1 or inplanes != out_channels:
            if shortcut_type == "B":
                self.downsample = nn.Sequential(
                    nn.Conv3d(inplanes, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_channels),
                )
            else:
                self.downsample = None
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is None:
            residual = _shortcut_a(x, self.out_channels, self.stride)
        else:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class MedicalNetResNetClassifier(nn.Module):
    def __init__(self, depth: int, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        if depth not in SUPPORTED_DEPTHS:
            raise ValueError(f"Unsupported depth {depth}. Choose from {sorted(SUPPORTED_DEPTHS)}")

        block_name, layers, shortcut_type = SUPPORTED_DEPTHS[depth]
        block = BasicBlock if block_name == "basic" else Bottleneck

        self.depth = depth
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_new_layers()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        layers = [block(self.inplanes, planes, stride=stride, shortcut_type=shortcut_type)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, shortcut_type=shortcut_type))
        return nn.Sequential(*layers)

    def _initialize_new_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.forward_features(x)
        return self.classifier(self.dropout(x))

    def set_encoder_trainable(self, trainable: bool):
        for name, param in self.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = trainable


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint does not contain a recognizable state_dict.")


def _clean_key(key: str) -> str:
    # Common wrappers from DataParallel and training containers.
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _adapt_conv1(weight: torch.Tensor, target_shape: torch.Size) -> Optional[torch.Tensor]:
    """Adapt a 1-channel pretrained conv1 to N input sequences."""
    if weight.ndim != 5 or len(target_shape) != 5:
        return None
    if weight.shape[0] != target_shape[0] or weight.shape[2:] != target_shape[2:]:
        return None
    old_c, new_c = weight.shape[1], target_shape[1]
    if old_c == new_c:
        return weight
    if old_c == 1 and new_c > 1:
        return weight.repeat(1, new_c, 1, 1, 1) / float(new_c)
    if new_c == 1 and old_c > 1:
        return weight.mean(dim=1, keepdim=True)
    return None


def load_medicalnet_pretrained(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    adapt_input_channels: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> dict:
    """Load only compatible encoder tensors; segmentation/classification heads are skipped.

    Returns a report containing loaded, skipped, and missing keys.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # older PyTorch
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    source = _extract_state_dict(checkpoint)
    target = model.state_dict()

    loaded = {}
    skipped = []

    for raw_key, value in source.items():
        key = _clean_key(raw_key)

        # Original MedicalNet segmentation head must never enter classifier weights.
        if key.startswith("conv_seg") or key.startswith("classifier") or key.startswith("fc"):
            skipped.append((raw_key, "head"))
            continue

        if key not in target:
            skipped.append((raw_key, "missing_key"))
            continue

        if value.shape == target[key].shape:
            loaded[key] = value
            continue

        if key == "conv1.weight" and adapt_input_channels:
            adapted = _adapt_conv1(value, target[key].shape)
            if adapted is not None:
                loaded[key] = adapted
                continue

        skipped.append((raw_key, f"shape {tuple(value.shape)} != {tuple(target[key].shape)}"))

    target.update(loaded)
    model.load_state_dict(target, strict=True)

    encoder_keys = [k for k in model.state_dict() if not k.startswith("classifier")]
    missing_encoder = [k for k in encoder_keys if k not in loaded]

    return {
        "checkpoint": str(checkpoint_path),
        "loaded_count": len(loaded),
        "loaded_keys": sorted(loaded),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "missing_encoder_count": len(missing_encoder),
        "missing_encoder_keys": missing_encoder,
    }

"""Faster R-CNN using the project's MobileNetV3-Large and custom FPN."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from models.backbone import MobileNetV3Backbone
from models.fpn import FeaturePyramidNetwork


CUSTOM_ARCHITECTURE = "custom_mobilenet_v3_large_fpn_v1"
LEGACY_ARCHITECTURE = "torchvision_mobilenet_v3_large_fpn"


class MobileNetV3WithFPN(nn.Module):
    """Join the hand-written backbone and FPN for Torchvision Faster R-CNN."""

    out_channels = 256

    def __init__(self) -> None:
        super().__init__()
        self.body = MobileNetV3Backbone()
        self.fpn = FeaturePyramidNetwork(
            in_channels=(40, 112, 160), out_channels=self.out_channels
        )

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return self.fpn(self.body(images))


def build_custom_detector(
    num_classes: int = 3,
    *,
    min_size: int = 800,
    max_size: int = 1333,
) -> FasterRCNN:
    """Build Faster R-CNN with custom C3/C4/C5 -> P3/P4/P5 features.

    Backbone, FPN, RPN and RoI heads all start from random initialization.
    """
    backbone = MobileNetV3WithFPN()
    anchor_generator = AnchorGenerator(
        sizes=((32, 64), (128, 256), (256, 512)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["P3", "P4", "P5"],
        output_size=7,
        sampling_ratio=2,
    )
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )


def build_legacy_detector(
    num_classes: int = 3,
    *,
    min_size: int = 800,
    max_size: int = 1333,
):
    """Build the former Torchvision model so existing checkpoints still load."""
    import torchvision

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_model(
    num_classes: int = 3,
    *,
    min_size: int = 800,
    max_size: int = 1333,
):
    """Build the custom detector with scratch-only initialization."""
    return build_custom_detector(
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
    )


def load_model(
    checkpoint: str | Path,
    device: torch.device,
    *,
    num_classes: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
):
    """Load new custom checkpoints and retain compatibility with old ones."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train first or pass --checkpoint."
        )
    state = torch.load(path, map_location=device, weights_only=True)
    architecture = (
        state.get("architecture", LEGACY_ARCHITECTURE)
        if isinstance(state, dict)
        else LEGACY_ARCHITECTURE
    )
    if architecture == CUSTOM_ARCHITECTURE:
        model = build_custom_detector(
            num_classes=num_classes, min_size=min_size, max_size=max_size
        )
    elif architecture == LEGACY_ARCHITECTURE:
        model = build_legacy_detector(
            num_classes=num_classes, min_size=min_size, max_size=max_size
        )
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")

    model_state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model

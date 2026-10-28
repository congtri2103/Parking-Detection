"""MobileNetV3-Large feature extractor for the custom Faster R-CNN tutorial.

This module intentionally exposes intermediate C3/C4/C5 feature maps instead
of returning classification logits.  It is the first building block of the
hand-built detector and is independent from ``models.detector`` (the complete
Torchvision detector used by the current training pipeline).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large


class MobileNetV3Backbone(nn.Module):
    """Extract C3, C4 and C5 maps from MobileNetV3-Large.

    For an input shaped ``[B, 3, 256, 384]`` the selected feature maps are:

    - C3: ``[B, 40, 32, 48]`` (stride 8)
    - C4: ``[B, 112, 16, 24]`` (stride 16)
    - C5: ``[B, 160, 8, 12]`` (stride 32)

    The layer selection follows the educational architecture in the project
    report.  It is not intended to reproduce Torchvision's internal FPN
    backbone exactly.
    """

    out_channels = {"C3": 40, "C4": 112, "C5": 160}
    strides = {"C3": 8, "C4": 16, "C5": 32}

    def __init__(self) -> None:
        super().__init__()
        # Explicitly forbid pretrained ImageNet weights.  This constructor does
        # not expose a pretrained switch, so scratch training cannot
        # accidentally trigger a weights download.
        self.features = mobilenet_v3_large(weights=None).features
        self.return_layers = {6: "C3", 12: "C4", 15: "C5"}
        self._last_layer = max(self.return_layers)

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "MobileNetV3Backbone expects [B, 3, H, W], "
                f"received {tuple(images.shape)}"
            )

        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        x = images
        for index, layer in enumerate(self.features):
            x = layer(x)
            name = self.return_layers.get(index)
            if name is not None:
                outputs[name] = x
            if index == self._last_layer:
                break
        return outputs

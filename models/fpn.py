"""A small, explicit Feature Pyramid Network implementation.

The implementation uses lateral 1x1 convolutions, nearest-neighbour top-down
upsampling with element-wise addition, and output 3x3 convolutions.  It does
not call ``torchvision.ops.FeaturePyramidNetwork``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class FeaturePyramidNetwork(nn.Module):
    """Convert C3/C4/C5 into same-width P3/P4/P5 feature maps."""

    input_names = ("C3", "C4", "C5")
    output_names = ("P3", "P4", "P5")

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (40, 112, 160),
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("in_channels must contain exactly C3, C4 and C5")
        if any(channels <= 0 for channels in in_channels) or out_channels <= 0:
            raise ValueError("all channel counts must be positive")

        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(channels, out_channels, kernel_size=1)
            for channels in in_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        )

    def forward(
        self, features: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in features]
        if missing:
            raise KeyError(f"FPN is missing input feature maps: {missing}")

        c3, c4, c5 = (features[name] for name in self.input_names)
        for name, feature, expected_channels in zip(
            self.input_names, (c3, c4, c5), self.in_channels
        ):
            if feature.ndim != 4 or feature.shape[1] != expected_channels:
                raise ValueError(
                    f"{name} must have shape [B, {expected_channels}, H, W], "
                    f"received {tuple(feature.shape)}"
                )

        l3 = self.lateral_convs[0](c3)
        l4 = self.lateral_convs[1](c4)
        l5 = self.lateral_convs[2](c5)

        m5 = l5
        m4 = l4 + F.interpolate(m5, size=l4.shape[-2:], mode="nearest")
        m3 = l3 + F.interpolate(m4, size=l3.shape[-2:], mode="nearest")

        return OrderedDict(
            P3=self.output_convs[0](m3),
            P4=self.output_convs[1](m4),
            P5=self.output_convs[2](m5),
        )


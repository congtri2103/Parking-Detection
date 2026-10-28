"""Shape tests for the hand-built MobileNetV3-Large backbone and FPN."""

from __future__ import annotations

import torch

from models.backbone import MobileNetV3Backbone
from models.detector import MobileNetV3WithFPN, get_model
from models.fpn import FeaturePyramidNetwork


EXPECTED_BACKBONE_SHAPES = {
    "C3": (1, 40, 32, 48),
    "C4": (1, 112, 16, 24),
    "C5": (1, 160, 8, 12),
}
EXPECTED_FPN_SHAPES = {
    "P3": (1, 256, 32, 48),
    "P4": (1, 256, 16, 24),
    "P5": (1, 256, 8, 12),
}


def assert_shapes(actual, expected) -> None:
    assert tuple(actual) == tuple(expected), f"expected {expected}, got {actual}"


def main() -> None:
    torch.manual_seed(0)
    backbone = MobileNetV3Backbone().eval()
    fpn = FeaturePyramidNetwork().eval()
    images = torch.randn(1, 3, 256, 384)

    with torch.inference_mode():
        features = backbone(images)
        pyramid = fpn(features)

    for name, expected in EXPECTED_BACKBONE_SHAPES.items():
        shape = tuple(features[name].shape)
        print(name, shape)
        assert_shapes(shape, expected)
    for name, expected in EXPECTED_FPN_SHAPES.items():
        shape = tuple(pyramid[name].shape)
        print(name, shape)
        assert_shapes(shape, expected)

    detector = get_model()
    assert isinstance(detector.backbone, MobileNetV3WithFPN)
    assert detector.backbone.body.return_layers == {6: "C3", 12: "C4", 15: "C5"}
    assert detector.backbone.out_channels == 256
    print("Detector backbone: custom MobileNetV3WithFPN (scratch)")

    print("PASS: backbone and FPN shape tests")


if __name__ == "__main__":
    main()

"""Run one custom-detector train step to validate wiring before a long run."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from models.detector import CUSTOM_ARCHITECTURE, get_model
from src.dataset import CocoVehicleDataset, collate_fn
from src.inference import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    dataset = CocoVehicleDataset(args.images, args.annotations)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    images, targets = next(iter(loader))
    images = [image.to(device) for image in images]
    targets = [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]
    model = get_model().to(device).train()

    with torch.autocast(device_type=device.type, enabled=args.amp):
        losses = model(images, targets)
        total = sum(losses.values())
    total.backward()

    print(f"Architecture: {CUSTOM_ARCHITECTURE}")
    print("Initialization: scratch")
    print(f"Device: {device}")
    for name, value in losses.items():
        print(f"{name}: {float(value.detach()):.6f}")
    print(f"total_loss: {float(total.detach()):.6f}")
    print("PASS: custom detector forward/backward")


if __name__ == "__main__":
    main()

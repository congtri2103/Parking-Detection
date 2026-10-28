"""Shared detector inference utilities."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def detect_frame(model, frame: np.ndarray, device: torch.device, threshold: float):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).to(device)
    prediction = model([tensor])[0]
    keep = prediction["scores"] >= threshold
    return (
        prediction["boxes"][keep].detach().cpu().numpy(),
        prediction["scores"][keep].detach().cpu().numpy(),
        prediction["labels"][keep].detach().cpu().numpy(),
    )


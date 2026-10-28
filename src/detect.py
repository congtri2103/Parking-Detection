"""Run the trained detector on one image and save an annotated image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from config import CLASS_NAMES
from models.detector import load_model
from src.inference import detect_frame, select_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", default="output/detection.jpg")
    parser.add_argument("--checkpoint", default="models/best.pth")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = cv2.imread(args.input)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    boxes, scores, labels = detect_frame(model, frame, device, args.confidence)
    for box, score, label in zip(boxes, scores, labels):
        if int(label) not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)
        cv2.putText(
            frame,
            f"{CLASS_NAMES[int(label)]} {score:.2f}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 205, 50),
            2,
            cv2.LINE_AA,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write image: {output}")
    print(f"Saved {len(boxes)} detections to {output}")


if __name__ == "__main__":
    main()


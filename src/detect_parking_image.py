"""Classify detected vehicles against an allowed-parking polygon in one image.

This is a spatial, image-only demo.  A single frame cannot prove that a vehicle
is stationary, so PARKING LEGAL/ILLEGAL means inside/outside the configured
parking area rather than a time-confirmed parking event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from config import CLASS_NAMES
from models.detector import load_model
from src.inference import detect_frame, select_device
from src.roi import bottom_center, load_roi, parse_roi, point_in_polygon


LEGAL_COLOR = (40, 190, 40)
ILLEGAL_COLOR = (30, 30, 230)
ROI_COLOR = (0, 210, 255)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output", default="output/demo_parking_result.jpg")
    parser.add_argument("--checkpoint", default="models/best.pth")
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument(
        "--allowed-roi", help="Allowed polygon as 'x,y;x,y;x,y;...'"
    )
    roi_group.add_argument("--allowed-roi-file", help="Allowed polygon JSON file")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def draw_label(frame, text: str, origin: tuple[int, int], color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y = origin
    x = max(0, min(x, frame.shape[1] - text_width - 8))
    y = max(text_height + 8, min(y, frame.shape[0] - baseline - 4))
    cv2.rectangle(
        frame,
        (x, y - text_height - 7),
        (x + text_width + 8, y + baseline + 3),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_allowed_area(frame, polygon: np.ndarray) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], LEGAL_COLOR)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [polygon], True, ROI_COLOR, 3, cv2.LINE_AA)
    x, y = map(int, polygon[0])
    draw_label(frame, "ALLOWED PARKING AREA", (x, max(28, y)), ROI_COLOR)


def draw_summary(frame, legal: int, illegal: int, total: int) -> None:
    lines = [
        f"DETECTED: {total}",
        f"PARKING LEGAL: {legal}",
        f"PARKING ILLEGAL: {illegal}",
    ]
    width = 285
    height = 34 + 30 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (16 + width, 16 + height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        color = (255, 255, 255)
        if index == 1:
            color = LEGAL_COLOR
        elif index == 2:
            color = ILLEGAL_COLOR
        cv2.putText(
            frame,
            line,
            (30, 50 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")

    frame = cv2.imread(args.input)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    roi = (
        parse_roi(args.allowed_roi)
        if args.allowed_roi
        else load_roi(args.allowed_roi_file)
    )
    polygon = np.asarray(roi, dtype=np.int32)

    device = select_device(args.device)
    model = load_model(args.checkpoint, device)
    boxes, scores, labels = detect_frame(model, frame, device, args.confidence)

    draw_allowed_area(frame, polygon)
    legal_count = 0
    illegal_count = 0
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        class_id = int(label)
        if class_id not in CLASS_NAMES:
            continue
        point = bottom_center(box)
        legal = point_in_polygon(point, roi)
        status = "PARKING LEGAL" if legal else "PARKING ILLEGAL"
        color = LEGAL_COLOR if legal else ILLEGAL_COLOR
        legal_count += int(legal)
        illegal_count += int(not legal)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.circle(frame, tuple(map(int, point)), 6, color, -1, cv2.LINE_AA)
        draw_label(
            frame,
            f"{CLASS_NAMES[class_id]} {float(score):.2f} | {status}",
            (x1, y1 - 6),
            color,
        )
        detections.append(
            {
                "class": CLASS_NAMES[class_id],
                "confidence": float(score),
                "box": [float(value) for value in box],
                "bottom_center": [float(point[0]), float(point[1])],
                "status": status,
            }
        )

    draw_summary(frame, legal_count, illegal_count, len(detections))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Could not write output image: {output}")

    summary = {
        "output": str(output),
        "device": str(device),
        "confidence_threshold": args.confidence,
        "detected": len(detections),
        "parking_legal": legal_count,
        "parking_illegal": illegal_count,
        "detections": detections,
    }
    report = output.with_suffix(".json")
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Image: {output}\nReport: {report}", flush=True)


if __name__ == "__main__":
    main()


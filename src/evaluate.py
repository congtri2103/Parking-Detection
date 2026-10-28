"""Evaluate the detector with per-class AP, precision, recall and latency."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CLASS_NAMES
from models.detector import load_model
from src.dataset import CocoVehicleDataset, collate_fn
from src.inference import select_device


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    box_area = np.prod(np.maximum(0.0, box[2:] - box[:2]))
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    return intersection / np.maximum(box_area + areas - intersection, 1e-9)


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP at one IoU threshold."""
    values = []
    for threshold in np.linspace(0, 1, 101):
        candidates = precision[recall >= threshold]
        values.append(float(candidates.max()) if candidates.size else 0.0)
    return float(np.mean(values))


def metrics_for_class(predictions, ground_truth, iou_threshold: float):
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in ground_truth.items()}
    ranked = sorted(predictions, key=lambda item: item[1], reverse=True)
    true_positive = np.zeros(len(ranked), dtype=np.float32)
    false_positive = np.zeros(len(ranked), dtype=np.float32)
    for index, (image_id, _score, box) in enumerate(ranked):
        targets = ground_truth.get(image_id, np.empty((0, 4), dtype=np.float32))
        ious = box_iou(box, targets)
        if ious.size:
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and not matched[image_id][best]:
                true_positive[index] = 1
                matched[image_id][best] = True
                continue
        false_positive[index] = 1
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / max(total_gt, 1)
    precision = tp / np.maximum(tp + fp, 1e-9)
    final_precision = float(precision[-1]) if precision.size else 0.0
    final_recall = float(recall[-1]) if recall.size else 0.0
    f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-9)
    return {
        "ground_truth": total_gt,
        "predictions": len(ranked),
        "precision": final_precision,
        "recall": final_recall,
        "f1": f1,
        "ap": interpolated_ap(recall, precision) if total_gt else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", default="models/best.pth")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dataset = CocoVehicleDataset(args.images, args.annotations)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = load_model(args.checkpoint, device)
    predictions = defaultdict(list)
    ground_truth = {class_id: {} for class_id in CLASS_NAMES}
    latencies = []

    for images, targets in tqdm(loader):
        image = images[0].to(device)
        target = targets[0]
        image_id = int(target["image_id"])
        for class_id in CLASS_NAMES:
            mask = target["labels"] == class_id
            ground_truth[class_id][image_id] = target["boxes"][mask].numpy()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model([image])[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if score >= args.score_threshold and int(label) in CLASS_NAMES:
                predictions[int(label)].append((image_id, float(score), box))

    class_metrics = {
        CLASS_NAMES[class_id]: metrics_for_class(
            predictions[class_id], ground_truth[class_id], args.iou_threshold
        )
        for class_id in CLASS_NAMES
    }
    summary = {
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "mAP": float(np.mean([item["ap"] for item in class_metrics.values()])),
        "mean_latency_ms": float(np.mean(latencies) * 1000),
        "fps": float(1.0 / np.mean(latencies)),
        "classes": class_metrics,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()


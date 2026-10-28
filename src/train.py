"""Train Faster R-CNN on COCO-format vehicle annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import CLASS_NAMES
from models.detector import CUSTOM_ARCHITECTURE, get_model
from src.dataset import CocoVehicleDataset, collate_fn
from src.evaluate import metrics_for_class
from src.inference import select_device


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    *,
    description: str,
) -> float:
    training = optimizer is not None
    model.train()
    if not training:
        # Detection models only return losses in train mode. Keep that mode for
        # validation loss while preventing validation images from updating the
        # custom MobileNet BatchNorm running statistics.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    total = 0.0
    amp_enabled = scaler is not None and scaler.is_enabled()
    progress = tqdm(
        loader,
        desc=description,
        leave=True,
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for step, (images, targets) in enumerate(progress, start=1):
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            loss = sum(model(images, targets).values())
        if training:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        loss_value = float(loss.detach())
        total += loss_value
        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            avg_loss=f"{total / step:.4f}",
            refresh=False,
        )
    average = total / max(1, len(loader))
    print(f"{description} loss: {average:.6f}", flush=True)
    return average


def evaluate_detection_metrics(
    model,
    loader,
    device,
    *,
    score_threshold: float,
    iou_threshold: float,
    description: str,
) -> dict:
    """Run validation inference and return aggregate and per-class metrics."""
    model.eval()
    predictions = defaultdict(list)
    ground_truth = {class_id: {} for class_id in CLASS_NAMES}

    progress = tqdm(
        loader,
        desc=description,
        leave=True,
        dynamic_ncols=True,
        mininterval=1.0,
    )
    with torch.inference_mode():
        for images, targets in progress:
            results = model([image.to(device) for image in images])
            for result, target in zip(results, targets):
                image_id = int(target["image_id"])
                for class_id in CLASS_NAMES:
                    mask = target["labels"] == class_id
                    ground_truth[class_id][image_id] = (
                        target["boxes"][mask].cpu().numpy()
                    )

                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    class_id = int(label)
                    if score >= score_threshold and class_id in CLASS_NAMES:
                        predictions[class_id].append(
                            (image_id, float(score), box)
                        )

    class_metrics = {
        CLASS_NAMES[class_id]: metrics_for_class(
            predictions[class_id], ground_truth[class_id], iou_threshold
        )
        for class_id in CLASS_NAMES
    }
    summary = {
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "mAP": float(np.mean([item["ap"] for item in class_metrics.values()])),
        "precision": float(
            np.mean([item["precision"] for item in class_metrics.values()])
        ),
        "recall": float(
            np.mean([item["recall"] for item in class_metrics.values()])
        ),
        "f1": float(np.mean([item["f1"] for item in class_metrics.values()])),
        "classes": class_metrics,
    }
    print(
        f"{description}: mAP@{iou_threshold:.2f}={summary['mAP']:.4f}, "
        f"precision={summary['precision']:.4f}, "
        f"recall={summary['recall']:.4f}, F1={summary['f1']:.4f}",
        flush=True,
    )
    for class_name, metrics in class_metrics.items():
        print(
            f"  {class_name}: AP={metrics['ap']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}",
            flush=True,
        )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--output", default="models/best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--metrics-score-threshold",
        type=float,
        default=0.05,
        help="Minimum score used for validation AP/precision/recall/F1",
    )
    parser.add_argument(
        "--metrics-iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold used for validation AP/precision/recall/F1",
    )
    parser.add_argument(
        "--motorcycle-weight",
        type=float,
        default=1.0,
        help="Sampling weight for train images containing motorcycles",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision to reduce GPU memory usage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    if args.motorcycle_weight < 1:
        raise ValueError("--motorcycle-weight must be >= 1")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    if not 0 <= args.metrics_score_threshold <= 1:
        raise ValueError("--metrics-score-threshold must be between 0 and 1")
    if not 0 < args.metrics_iou_threshold <= 1:
        raise ValueError("--metrics-iou-threshold must be in (0, 1]")
    train_data = CocoVehicleDataset(args.train_images, args.train_annotations)
    val_data = CocoVehicleDataset(args.val_images, args.val_annotations)
    sampler = None
    if args.motorcycle_weight > 1:
        sample_weights = [
            args.motorcycle_weight if train_data.has_class(i, 2) else 1.0
            for i in range(len(train_data))
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    model = get_model().to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    best_map = float("-inf")
    best_loss_output = output.with_name(f"{output.stem}_loss{output.suffix}")

    print(
        f"Training on {device}; train={len(train_data)}, val={len(val_data)}, "
        f"amp={args.amp}, motorcycle_weight={args.motorcycle_weight}, "
        f"architecture={CUSTOM_ARCHITECTURE}, "
        "initialization=scratch (weights=None)"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            description=f"Epoch {epoch}/{args.epochs} train",
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                device,
                scaler=scaler,
                description=f"Epoch {epoch}/{args.epochs} val",
            )
        metrics = evaluate_detection_metrics(
            model,
            val_loader,
            device,
            score_threshold=args.metrics_score_threshold,
            iou_threshold=args.metrics_iou_threshold,
            description=f"Epoch {epoch}/{args.epochs} metrics",
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "metrics": metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        output.with_suffix(".history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": CUSTOM_ARCHITECTURE,
                    "initialization": "scratch",
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "selection_metric": "val_loss",
                    "class_names": {1: "car", 2: "motorcycle"},
                },
                best_loss_output,
            )
            print(f"Saved best-loss checkpoint to {best_loss_output}", flush=True)
        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": CUSTOM_ARCHITECTURE,
                    "initialization": "scratch",
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "selection_metric": "mAP@IoU",
                    "class_names": {1: "car", 2: "motorcycle"},
                },
                output,
            )
            print(f"Saved best-mAP checkpoint to {output}", flush=True)


if __name__ == "__main__":
    main()

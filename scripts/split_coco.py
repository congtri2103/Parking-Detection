#!/usr/bin/env python3
"""Create deterministic train/val/test COCO splits without copying images."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Source image directory; required when materializing split images",
    )
    parser.add_argument(
        "--materialize",
        choices=("none", "hardlink", "copy", "symlink"),
        default="none",
        help="Populate data/<split>/images; hardlink avoids duplicate disk usage",
    )
    return parser.parse_args()


def allocate(bucket: list[dict], val_ratio: float, test_ratio: float):
    size = len(bucket)
    val_size = round(size * val_ratio)
    test_size = round(size * test_ratio)
    return {
        "val": bucket[:val_size],
        "test": bucket[val_size : val_size + test_size],
        "train": bucket[val_size + test_size :],
    }


def materialize_images(
    images: list[dict], source_dir: Path, destination_dir: Path, mode: str
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        source = (source_dir / image["file_name"]).resolve()
        destination = destination_dir / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        if destination.exists() or destination.is_symlink():
            continue
        if mode == "hardlink":
            os.link(source, destination)
        elif mode == "copy":
            shutil.copy2(source, destination)
        elif mode == "symlink":
            destination.symlink_to(source)


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("Split ratios cannot be negative")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val-ratio + test-ratio must be less than 1")
    if args.materialize != "none" and args.images_dir is None:
        raise ValueError("--images-dir is required when --materialize is enabled")

    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = source.get("images", [])
    annotations = source.get("annotations", [])
    if not images:
        raise ValueError("COCO file contains no images")

    category_names = {
        int(category["id"]): str(category["name"]).strip().lower()
        for category in source.get("categories", [])
    }
    image_classes: dict[int, set[str]] = defaultdict(set)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)
        name = category_names.get(int(annotation["category_id"]))
        if name:
            image_classes[image_id].add(name)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for image in images:
        classes = image_classes[int(image["id"])]
        key = (
            "motorcycle"
            if "motorcycle" in classes or "motorbike" in classes
            else "car"
        )
        buckets[key].append(image)

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
        allocated = allocate(buckets[key], args.val_ratio, args.test_ratio)
        for split_name in splits:
            splits[split_name].extend(allocated[split_name])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    for split_name, split_images in splits.items():
        rng.shuffle(split_images)
        image_ids = {int(image["id"]) for image in split_images}
        if seen & image_ids:
            raise RuntimeError("An image was assigned to more than one split")
        seen.update(image_ids)
        split_annotations = [
            annotation
            for image_id in image_ids
            for annotation in annotations_by_image.get(image_id, [])
        ]
        payload = {
            "info": {
                **source.get("info", {}),
                "split": split_name,
                "split_seed": args.seed,
            },
            "images": split_images,
            "annotations": split_annotations,
            "categories": source.get("categories", []),
        }
        destination = args.output_dir / f"{split_name}.json"
        destination.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        class_images = Counter(
            "motorcycle"
            if "motorcycle" in image_classes[int(image["id"])]
            else "car_only"
            for image in split_images
        )
        print(
            f"{split_name}: images={len(split_images)}, "
            f"annotations={len(split_annotations)}, {dict(class_images)}"
        )
        if args.materialize != "none":
            materialize_images(
                split_images,
                args.images_dir,
                args.output_dir.parent / split_name / "images",
                args.materialize,
            )

    if seen != {int(image["id"]) for image in images}:
        raise RuntimeError("Some images were not assigned to a split")


if __name__ == "__main__":
    main()

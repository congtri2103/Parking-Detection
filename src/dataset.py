"""COCO-format dataset for car/motorcycle Faster R-CNN training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


SUPPORTED_CATEGORIES = {
    "car": 1,
    "motorcycle": 2,
    "motorbike": 2,
}


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return F.convert_image_dtype(F.pil_to_tensor(image), torch.float32)


class CocoVehicleDataset(Dataset):
    """Read a small COCO detection dataset without requiring pycocotools.

    COCO category IDs are remapped by category *name*, so input IDs may be any
    positive integers. Unsupported categories are ignored.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations)
        self.transform = transform
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.annotations_path}")

        data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        category_map = {
            int(item["id"]): SUPPORTED_CATEGORIES[item["name"].strip().lower()]
            for item in data.get("categories", [])
            if item.get("name", "").strip().lower() in SUPPORTED_CATEGORIES
        }
        if not category_map:
            raise ValueError("COCO file has no 'car' or 'motorcycle' categories")

        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in data.get("annotations", []):
            if int(annotation.get("category_id", -1)) in category_map:
                annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
                    annotation
                )
        self.images = sorted(data.get("images", []), key=lambda item: int(item["id"]))
        if not self.images:
            raise ValueError("COCO file contains no images")
        self.annotations_by_image = annotations_by_image
        self.category_map = category_map

    def has_class(self, index: int, class_id: int) -> bool:
        """Return whether an image contains at least one remapped class."""
        image_id = int(self.images[index]["id"])
        return any(
            self.category_map[int(annotation["category_id"])] == class_id
            for annotation in self.annotations_by_image.get(image_id, [])
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        info = self.images[index]
        image_id = int(info["id"])
        path = self.images_dir / info["file_name"]
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        crowds: list[int] = []
        for annotation in self.annotations_by_image.get(image_id, []):
            x, y, width, height = map(float, annotation["bbox"])
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_map[int(annotation["category_id"])])
            areas.append(float(annotation.get("area", width * height)))
            crowds.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = image_to_tensor(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))

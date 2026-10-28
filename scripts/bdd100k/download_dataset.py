#!/usr/bin/env python3
"""
Download the Hugging Face mirror dgural/bdd100k (10,000-sample BDD100K
validation dataset via FiftyOne), keep only car/motorcycle detections,
and export a COCO-style dataset.

Output:
dataset_10k/
  images/
  annotations/instances.json

Important:
- This source contains 10,000 source images total.
- After filtering to images that contain car and/or motorcycle, the final
  number can be LESS than 10,000. The script reports the real count.
- Other object annotations are discarded; only car and motorcycle remain.
"""

from pathlib import Path
import argparse, json, shutil

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="dataset_10k")
    p.add_argument("--max-samples", type=int, default=10000)
    args = p.parse_args()

    try:
        import fiftyone as fo
        import fiftyone.utils.huggingface as fouh
    except ImportError:
        raise SystemExit(
            "Missing FiftyOne. Install with:\n"
            "python3 -m pip install -U fiftyone"
        )

    out = Path(args.output).resolve()
    image_dir = out / "images"
    ann_dir = out / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading dgural/bdd100k ...")
    ds = fouh.load_from_hub(
        "dgural/bdd100k",
        max_samples=args.max_samples,
        overwrite=False,
    )

    categories = [
        {"id": 1, "name": "car", "supercategory": "vehicle"},
        {"id": 2, "name": "motorcycle", "supercategory": "vehicle"},
    ]
    cat_id = {"car": 1, "motorcycle": 2}

    images = []
    annotations = []
    ann_id = 1
    image_id = 1
    class_boxes = {"car": 0, "motorcycle": 0}
    class_images = {"car": 0, "motorcycle": 0}

    for sample in ds:
        dets = getattr(sample, "detections", None)
        if dets is None:
            continue

        kept = [
            d for d in dets.detections
            if str(d.label).lower() in ("car", "motorcycle")
        ]
        if not kept:
            continue

        src = Path(sample.filepath)
        if not src.exists():
            continue

        # Metadata is usually available; compute if needed.
        md = sample.metadata
        if md is None or not md.width or not md.height:
            sample.compute_metadata()
            md = sample.metadata
        w, h = int(md.width), int(md.height)

        # Prefix with image id to avoid any filename collision.
        dst_name = f"{image_id:05d}_{src.name}"
        shutil.copy2(src, image_dir / dst_name)

        images.append({
            "id": image_id,
            "file_name": dst_name,
            "width": w,
            "height": h,
        })

        present = set()
        for d in kept:
            label = str(d.label).lower()
            # FiftyOne Detection bounding_box = normalized [x, y, width, height]
            x, y, bw, bh = d.bounding_box
            x, y, bw, bh = x*w, y*h, bw*w, bh*h

            # Clip defensively.
            x = max(0.0, min(x, w - 1))
            y = max(0.0, min(y, h - 1))
            bw = max(0.0, min(bw, w - x))
            bh = max(0.0, min(bh, h - y))
            if bw <= 1 or bh <= 1:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id[label],
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1
            class_boxes[label] += 1
            present.add(label)

        for label in present:
            class_images[label] += 1
        image_id += 1

    coco = {
        "info": {
            "description": "BDD100K subset filtered to car and motorcycle",
            "source": "dgural/bdd100k on Hugging Face",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    ann_file = ann_dir / "instances.json"
    with ann_file.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print("\n========== DONE ==========")
    print(f"Source samples checked : {len(ds)}")
    print(f"Images kept            : {len(images)}")
    print(f"Car images             : {class_images['car']}")
    print(f"Motorcycle images      : {class_images['motorcycle']}")
    print(f"Car boxes              : {class_boxes['car']}")
    print(f"Motorcycle boxes       : {class_boxes['motorcycle']}")
    print(f"Images directory       : {image_dir}")
    print(f"COCO annotations       : {ann_file}")
    print("==========================")
    if len(images) < 10000:
        print(
            "\nNOTE: Fewer than 10,000 images is expected if some source images "
            "contain neither car nor motorcycle. Do not duplicate images just "
            "to force the count to 10,000."
        )

if __name__ == "__main__":
    main()

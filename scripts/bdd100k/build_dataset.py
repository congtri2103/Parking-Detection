#!/usr/bin/env python3
"""
Build a 10,000-image, 2-class car/motorcycle detection dataset from BDD100K.

Expected BDD100K layout:
bdd100k/
├── images/100k/train/*.jpg
├── images/100k/val/*.jpg
└── labels/det_20/
    ├── det_train.json
    └── det_val.json

Output:
dataset_10k/
├── train/images/          # 8,000 images
├── val/images/            # 1,000 images
├── test/images/           # 1,000 images
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json

Only annotations for:
1 = car
2 = motorcycle

Selection target:
- 7,000 selected images containing at least one car
- 3,000 selected images containing at least one motorcycle
Images with both classes are assigned to the scarcer quota first.
"""
import argparse, json, random, shutil
from pathlib import Path
from collections import Counter

TARGET = {"car": 7000, "motorcycle": 3000}
SPLITS = {"train": 8000, "val": 1000, "test": 1000}
CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "motorcycle", "supercategory": "vehicle"},
]

def load_frames(label_file, image_dir):
    with open(label_file, "r", encoding="utf-8") as f:
        frames = json.load(f)
    out = []
    for fr in frames:
        labels = fr.get("labels") or []
        anns = []
        present = set()
        for lab in labels:
            cat = lab.get("category")
            if cat not in ("car", "motorcycle"):
                continue
            b = lab.get("box2d")
            if not b:
                continue
            x1,y1,x2,y2 = map(float, (b["x1"],b["y1"],b["x2"],b["y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            anns.append((cat, [x1,y1,x2-x1,y2-y1]))
            present.add(cat)
        p = image_dir / fr["name"]
        if anns and p.exists():
            out.append({
                "path": p, "name": fr["name"], "anns": anns,
                "present": present, "attrs": fr.get("attributes", {})
            })
    return out

def choose_10k(items, seed):
    rng = random.Random(seed)
    moto = [x for x in items if "motorcycle" in x["present"]]
    car_only = [x for x in items if "car" in x["present"] and "motorcycle" not in x["present"]]
    both = [x for x in moto if "car" in x["present"]]
    moto_only = [x for x in moto if "car" not in x["present"]]
    rng.shuffle(moto_only); rng.shuffle(both); rng.shuffle(car_only)

    # First satisfy 3k motorcycle-image target.
    selected_m = (moto_only + both)[:TARGET["motorcycle"]]
    used = {str(x["path"]) for x in selected_m}

    # Then satisfy remaining total with car-containing images.
    car_pool = [x for x in car_only + both if str(x["path"]) not in used]
    rng.shuffle(car_pool)
    selected_c = car_pool[:TARGET["car"]]
    selected = selected_m + selected_c

    # If overlap/availability leaves us short, fill from any eligible unused image.
    if len(selected) < 10000:
        rest = [x for x in items if str(x["path"]) not in {str(y["path"]) for y in selected}]
        rng.shuffle(rest)
        selected += rest[:10000-len(selected)]

    if len(selected) < 10000:
        raise RuntimeError(
            f"Only {len(selected)} eligible images found. "
            "Use full BDD100K detection train+val images/labels."
        )
    rng.shuffle(selected)
    return selected[:10000]

def stratified_split(items, seed):
    # Lightweight deterministic balancing by class-presence + time-of-day where available.
    rng = random.Random(seed)
    buckets = {}
    for x in items:
        key = (
            "both" if len(x["present"]) == 2 else next(iter(x["present"])),
            x["attrs"].get("timeofday", "undefined")
        )
        buckets.setdefault(key, []).append(x)
    for v in buckets.values(): rng.shuffle(v)

    result = {"train": [], "val": [], "test": []}
    desired = SPLITS.copy()
    # Round-robin from each bucket with 80/10/10 preference.
    all_items = []
    for _, bucket in sorted(buckets.items()):
        all_items.extend(bucket)
    rng.shuffle(all_items)
    for x in all_items:
        ratios = {
            k: len(result[k]) / desired[k] if desired[k] else 1
            for k in result
        }
        k = min((s for s in result if len(result[s]) < desired[s]),
                key=lambda s: ratios[s])
        result[k].append(x)
    return result

def write_coco(split_name, items, out_root):
    img_out = out_root / split_name / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out = out_root / "annotations"
    ann_out.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    for image_id, item in enumerate(items, 1):
        dst = img_out / item["name"]
        shutil.copy2(item["path"], dst)

        # BDD100K detection images are normally 1280x720.
        # Read actual dimensions if Pillow is installed.
        try:
            from PIL import Image
            with Image.open(item["path"]) as im:
                w,h = im.size
        except Exception:
            w,h = 1280,720

        images.append({
            "id": image_id, "file_name": item["name"],
            "width": w, "height": h
        })
        for cat, bbox in item["anns"]:
            x,y,bw,bh = bbox
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1 if cat == "car" else 2,
                "bbox": [x,y,bw,bh],
                "area": bw*bh,
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "info": {"description": "BDD100K filtered: car + motorcycle only"},
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }
    with open(ann_out / f"instances_{split_name}.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd-root", required=True, type=Path)
    ap.add_argument("--output", default=Path("dataset_10k"), type=Path)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    root = args.bdd_root
    candidates = []
    for split in ("train", "val"):
        labels = root / "labels" / "det_20" / f"det_{split}.json"
        imgs = root / "images" / "100k" / split
        if labels.exists() and imgs.exists():
            candidates.extend(load_frames(labels, imgs))

    print("Eligible images:", len(candidates))
    selected = choose_10k(candidates, args.seed)
    parts = stratified_split(selected, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, subset in parts.items():
        write_coco(name, subset, args.output)
        counts = Counter()
        for x in subset:
            counts.update(x["present"])
        print(name, len(subset), dict(counts))

    print("\nDone:", args.output.resolve())
    print("Classes: 1=car, 2=motorcycle")
    print("Split: 8000 train / 1000 val / 1000 test")

if __name__ == "__main__":
    main()

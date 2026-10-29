import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/airborne_dataset/MBAOD")
SPLIT = "Train - V1 - Clean"

coco_path = ROOT / "Annotations" / "COCO" / SPLIT / "_annotations.coco.json"
yolo_dir = ROOT / "Annotations" / "YOLOv8" / SPLIT / "labels"

data = json.loads(coco_path.read_text())
categories = {c["id"]: c["name"] for c in data["categories"]}
images = {img["id"]: img for img in data["images"]}

# Index COCO annotations by image filename.
coco_by_file = defaultdict(list)
for ann in data["annotations"]:
    img = images.get(ann["image_id"])
    if img:
        coco_by_file[img["file_name"]].append(ann)

print("COCO categories:")
for cid, name in sorted(categories.items()):
    print(f"  {cid}: {name}")

print("\nYOLO class ID -> filename-prefix counts:")
prefix_counts = defaultdict(Counter)

for path in yolo_dir.glob("*.txt"):
    prefix = path.name.split("_")[0]
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                prefix_counts[int(parts[0])][prefix] += 1
            except ValueError:
                pass

for cid, counts in sorted(prefix_counts.items()):
    print(f"  {cid}: {counts.most_common(5)}")

print("\nCOCO category counts:")
coco_counts = Counter(categories[ann["category_id"]] for ann in data["annotations"])
for name, count in sorted(coco_counts.items()):
    print(f"  {name}: {count}")

print("\nSample COCO image records:")
for img in list(data["images"])[:3]:
    print(" ", img["file_name"], "->", len(coco_by_file[img["file_name"]]), "annotations")

print("\nNote: this compares category statistics; it does not yet prove"
      " that every YOLO box matches its COCO counterpart.")

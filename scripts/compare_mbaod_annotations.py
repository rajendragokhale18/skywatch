import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/airborne_dataset/MBAOD")
SPLIT = "Train - V1 - Clean"

YOLO_DIR = ROOT / "Annotations" / "YOLOv8" / SPLIT / "labels"
COCO_FILE = ROOT / "Annotations" / "COCO" / SPLIT / "_annotations.coco.json"

OUTPUT = Path("results/mbaod_annotation_mismatches.txt")
IOU_THRESHOLD = 0.50

# Provisional mapping, supported by comparison with COCO annotations.
YOLO_NAMES = {
    0: "airplane",
    1: "bird",
    2: "helicopter",
    3: "kite",
    4: "parachute",
    5: "uav",
}


def calculate_iou(box_a, box_b):
    """Calculate IoU for two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_coco_annotations():
    """Load COCO metadata and index annotations by image stem."""
    data = json.loads(COCO_FILE.read_text())

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    images_by_id = {
        image["id"]: image
        for image in data["images"]
    }

    images_by_stem = {
        Path(image["file_name"]).stem: image
        for image in data["images"]
    }

    annotations_by_stem = defaultdict(list)

    for annotation in data["annotations"]:
        image = images_by_id.get(annotation["image_id"])
        if image is not None:
            stem = Path(image["file_name"]).stem
            annotations_by_stem[stem].append(annotation)

    return categories, images_by_stem, annotations_by_stem


def read_yolo_boxes(label_path, image_width, image_height):
    """Convert normalized YOLO boxes into pixel-coordinate boxes."""
    boxes = []
    malformed_lines = 0

    for line_number, line in enumerate(
        label_path.read_text().splitlines(), start=1
    ):
        parts = line.split()

        if not parts:
            continue

        if len(parts) != 5:
            malformed_lines += 1
            continue

        try:
            class_id = int(parts[0])
            xc, yc, width, height = map(float, parts[1:])
        except ValueError:
            malformed_lines += 1
            continue

        if (
            class_id not in YOLO_NAMES
            or not all(
                0.0 <= value <= 1.0
                for value in (xc, yc, width, height)
            )
            or width <= 0
            or height <= 0
        ):
            malformed_lines += 1
            continue

        box = [
            (xc - width / 2) * image_width,
            (yc - height / 2) * image_height,
            (xc + width / 2) * image_width,
            (yc + height / 2) * image_height,
        ]

        boxes.append((class_id, box))

    return boxes, malformed_lines


def read_coco_boxes(annotations):
    """Convert COCO xywh boxes into pixel-coordinate xyxy boxes."""
    boxes = []

    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        boxes.append(
            (
                annotation["category_id"],
                [x, y, x + width, y + height],
            )
        )

    return boxes


def match_boxes(yolo_boxes, coco_boxes):
    """
    Greedy one-to-one matching by descending IoU.

    Geometry is matched first; class agreement is checked afterward.
    """
    candidates = []

    for yolo_index, (_, yolo_box) in enumerate(yolo_boxes):
        for coco_index, (_, coco_box) in enumerate(coco_boxes):
            score = calculate_iou(yolo_box, coco_box)

            if score >= IOU_THRESHOLD:
                candidates.append((score, yolo_index, coco_index))

    used_yolo = set()
    used_coco = set()
    matches = []

    for score, yolo_index, coco_index in sorted(
        candidates, reverse=True
    ):
        if yolo_index in used_yolo or coco_index in used_coco:
            continue

        used_yolo.add(yolo_index)
        used_coco.add(coco_index)

        matches.append((score, yolo_index, coco_index))

    unmatched_yolo = [
        index
        for index in range(len(yolo_boxes))
        if index not in used_yolo
    ]

    unmatched_coco = [
        index
        for index in range(len(coco_boxes))
        if index not in used_coco
    ]

    return matches, unmatched_yolo, unmatched_coco


def main():
    if not COCO_FILE.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {COCO_FILE}")

    if not YOLO_DIR.is_dir():
        raise FileNotFoundError(f"YOLO labels directory not found: {YOLO_DIR}")

    categories, images_by_stem, annotations_by_stem = (
        load_coco_annotations()
    )

    stats = Counter()
    class_pairs = Counter()
    mismatches = []

    for label_path in sorted(YOLO_DIR.glob("*.txt")):
        stem = label_path.stem
        image = images_by_stem.get(stem)

        if image is None:
            stats["labels_without_coco_image"] += 1
            mismatches.append(
                f"NO COCO IMAGE | {stem} | label={label_path}"
            )
            continue

        yolo_boxes, malformed = read_yolo_boxes(
            label_path, image["width"], image["height"]
        )
        coco_boxes = read_coco_boxes(
            annotations_by_stem.get(stem, [])
        )

        stats["images_compared"] += 1
        stats["yolo_boxes"] += len(yolo_boxes)
        stats["coco_boxes"] += len(coco_boxes)
        stats["malformed_lines"] += malformed

        matches, unmatched_yolo, unmatched_coco = match_boxes(
            yolo_boxes, coco_boxes
        )

        stats["matched_boxes"] += len(matches)

        for score, yi, ci in matches:
            yolo_id, _ = yolo_boxes[yi]
            coco_id, _ = coco_boxes[ci]

            yolo_name = YOLO_NAMES[yolo_id]
            coco_name = categories.get(coco_id, f"unknown_{coco_id}")

            class_pairs[(yolo_name, coco_name)] += 1

            if yolo_name != coco_name:
                stats["class_disagreements"] += 1
                mismatches.append(
                    f"CLASS DISAGREEMENT | {stem} | "
                    f"YOLO={yolo_name} | COCO={coco_name} | "
                    f"IoU={score:.4f}"
                )

        for index in unmatched_yolo:
            class_id, box = yolo_boxes[index]
            stats["unmatched_yolo_boxes"] += 1
            mismatches.append(
                f"YOLO UNMATCHED | {stem} | "
                f"class={class_id} ({YOLO_NAMES[class_id]}) | "
                f"box={[round(value, 2) for value in box]}"
            )

        for index in unmatched_coco:
            class_id, box = coco_boxes[index]
            stats["unmatched_coco_boxes"] += 1
            class_name = categories.get(class_id, f"unknown_{class_id}")
            mismatches.append(
                f"COCO UNMATCHED | {stem} | "
                f"category={class_id} ({class_name}) | "
                f"box={[round(value, 2) for value in box]}"
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "\n".join(mismatches) + ("\n" if mismatches else "")
    )

    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Images compared: {stats['images_compared']}")
    print(f"YOLO boxes: {stats['yolo_boxes']}")
    print(f"COCO boxes: {stats['coco_boxes']}")
    print(f"Matched boxes: {stats['matched_boxes']}")
    print(f"Unmatched YOLO boxes: {stats['unmatched_yolo_boxes']}")
    print(f"Unmatched COCO boxes: {stats['unmatched_coco_boxes']}")
    print(f"Class disagreements: {stats['class_disagreements']}")
    print(f"Malformed YOLO lines: {stats['malformed_lines']}")
    print(f"Labels without COCO image: {stats['labels_without_coco_image']}")

    print("\nMatched class pairs (YOLO -> COCO):")
    for pair, count in class_pairs.most_common():
        print(f"  {pair[0]} -> {pair[1]}: {count}")

    print(f"\nMismatch details saved to: {OUTPUT}")
    print("\nFirst 15 mismatch records:")

    for record in mismatches[:15]:
        print(record)

    if not mismatches:
        print("No mismatches found.")


if __name__ == "__main__":
    main()

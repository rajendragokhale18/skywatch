
from collections import Counter
from pathlib import Path

ROOT = Path(
    "data/airborne_dataset/MBAOD"
)

IMAGE_ROOT = ROOT / "Images"
LABEL_ROOT = ROOT / "Annotations" / "YOLOv8"

SPLITS = {
    "train": "Train - V1 - Clean",
    "valid": "Vaild",
    "test_clean": "Test - Clean",
    "test_weather": "Test - Adverse Weather Conditions",
}


def main():
    for split_name, folder_name in SPLITS.items():
        image_dir = IMAGE_ROOT / folder_name
        label_dir = LABEL_ROOT / folder_name / "labels"

        images = []
        if image_dir.is_dir():
            images = [
                path for path in image_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]

        labels = list(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        label_names = {path.stem for path in labels}
        image_names = {path.stem for path in images}

        class_counts = Counter()
        malformed = 0
        empty_labels = 0

        for label_path in labels:
            lines = label_path.read_text().splitlines()

            if not lines:
                empty_labels += 1

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    malformed += 1
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                    if not all(0 <= value <= 1 for value in coords):
                        malformed += 1
                        continue
                    class_counts[class_id] += 1
                except ValueError:
                    malformed += 1

        print(f"\n--- {split_name} ---")
        print(f"Images: {len(images)}")
        print(f"Label files: {len(labels)}")
        print(f"Images without matching label file: {len(image_names - label_names)}")
        print(f"Labels without matching image: {len(label_names - image_names)}")
        print(f"Empty label files: {empty_labels}")
        print(f"Malformed annotation lines: {malformed}")
        print(f"Class ID counts: {dict(sorted(class_counts.items()))}")


if __name__ == "__main__":
    main()

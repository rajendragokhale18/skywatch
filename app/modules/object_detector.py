
from pathlib import Path

import cv2
from ultralytics import YOLO


class ObjectDetector:
    """YOLO detector with optional tiled inference."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.25,
        image_size: int = 640,
        tile_size: int = 640,
        overlap: float = 0.20,
    ):
        if not 0 < overlap < 1:
            raise ValueError("overlap must be between 0 and 1.")
        if tile_size < 64:
            raise ValueError("tile_size must be at least 64.")

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.tile_size = tile_size
        self.overlap = overlap

    def _predict(self, image):
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0].cpu()),
                "class_id": int(box.cls[0].cpu()),
                "class_name": self.model.names[
                    int(box.cls[0].cpu())
                ],
            })

        return detections

    def detect(self, image, tiled: bool = True):
        """Return image-coordinate detections with bounding boxes."""

        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        height, width = image.shape[:2]

        if not tiled or (
            width <= self.tile_size and height <= self.tile_size
        ):
            return self._predict(image)

        step = max(1, int(self.tile_size * (1 - self.overlap)))
        candidates = []

        for y in range(0, height, step):
            for x in range(0, width, step):
                x2 = min(x + self.tile_size, width)
                y2 = min(y + self.tile_size, height)
                x1 = max(0, x2 - self.tile_size)
                y1 = max(0, y2 - self.tile_size)

                tile = image[y1:y2, x1:x2]
                if tile.size == 0:
                    continue

                for detection in self._predict(tile):
                    bx1, by1, bx2, by2 = detection["bbox"]
                    detection["bbox"] = [
                        bx1 + x1,
                        by1 + y1,
                        bx2 + x1,
                        by2 + y1,
                    ]
                    candidates.append(detection)

        return self._nms(candidates, iou_threshold=0.50)

    @staticmethod
    def _nms(
        detections,
        iou_threshold=0.50,
        class_agnostic=False,
    ):
        """Apply NMS, optionally suppressing boxes across classes."""

        if not detections:
            return []

        if class_agnostic:
            groups = [detections]
        else:
            class_ids = sorted({d["class_id"] for d in detections})
            groups = [
                [d for d in detections if d["class_id"] == class_id]
                for class_id in class_ids
            ]

        kept = []

        for group in groups:
            items = sorted(
                group,
                key=lambda d: d["confidence"],
                reverse=True,
            )

            while items:
                best = items.pop(0)
                kept.append(best)
                bx1, by1, bx2, by2 = best["bbox"]
                remaining = []

                for item in items:
                    x1, y1, x2, y2 = item["bbox"]
                    ix1 = max(bx1, x1)
                    iy1 = max(by1, y1)
                    ix2 = min(bx2, x2)
                    iy2 = min(by2, y2)

                    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_best = max(0, bx2 - bx1) * max(0, by2 - by1)
                    area_item = max(0, x2 - x1) * max(0, y2 - y1)
                    union = area_best + area_item - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou < iou_threshold:
                        remaining.append(item)

                items = remaining

        return sorted(kept, key=lambda d: d["confidence"], reverse=True)

def detect_image(
    input_path: str,
    output_path: str,
    model_path: str = "yolo26n.pt",
    tiled: bool = True,
):
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not read image: {source}")

    detector = ObjectDetector(model_path=model_path)
    detections = detector.detect(image, tiled=tiled)

    for item in detections:
        x1, y1, x2, y2 = map(int, item["bbox"])
        label = f"{item['class_name']} {item['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(destination), image):
        raise RuntimeError(f"Could not save output: {destination}")

    print(f"Detections: {len(detections)}")
    print(f"Saved annotated image: {destination}")
    for item in detections:
        print(item)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--no-tiling", action="store_true")
    args = parser.parse_args()

    detect_image(
        args.input,
        args.output,
        model_path=args.model,
        tiled=not args.no_tiling,
    )

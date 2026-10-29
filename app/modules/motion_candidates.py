
from collections import defaultdict, deque
from pathlib import Path
import csv

import cv2
import numpy as np

from app.modules.tracker import CentroidTracker
from app.modules.motion_features import MotionFeatureExtractor


def detect_motion_video(
    input_path: str,
    output_path: str,
    min_area: int = 80,
    features_csv_path: str = "results/motion_features.csv",
) -> None:
    source = Path(input_path)
    destination = Path(output_path)
    csv_destination = Path(features_csv_path)

    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video has invalid FPS or dimensions.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {destination}")

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=25,
        detectShadows=False,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    tracker = CentroidTracker(max_disappeared=10, max_distance=60.0)
    extractor = MotionFeatureExtractor(history_size=30, fps=fps)
    trails = defaultdict(lambda: deque(maxlen=30))

    fields = [
        "frame",
        "timestamp_sec",
        "object_id",
        "centroid_x",
        "centroid_y",
        "observations",
        "speed_px_per_sec",
        "straightness",
        "hover_ratio",
        "direction_change_deg",
    ]

    frame_count = 0
    detection_count = 0
    csv_rows = 0

    try:
        with csv_destination.open("w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                mask = background_subtractor.apply(frame)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                boxes = []
                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue

                    boxes.append(cv2.boundingRect(contour))
                    detection_count += 1

                tracked_objects = tracker.update(boxes)

                for object_id, centroid in tracked_objects.items():
                    features = extractor.update(object_id, centroid)
                    trails[object_id].append(centroid)

                    x, y = centroid
                    cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"ID {object_id} | {features['speed_px_per_sec']:.1f}px/s",
                        (x + 8, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

                    points = list(trails[object_id])
                    for i in range(1, len(points)):
                        cv2.line(
                            frame,
                            points[i - 1],
                            points[i],
                            (255, 200, 0),
                            2,
                        )

                    csv_writer.writerow({
                        "frame": frame_count,
                        "timestamp_sec": round(frame_count / fps, 3),
                        "object_id": object_id,
                        "centroid_x": x,
                        "centroid_y": y,
                        **features,
                    })
                    csv_rows += 1

                cv2.putText(
                    frame,
                    f"Frame: {frame_count} | Active tracks: {len(tracked_objects)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                writer.write(frame)
                frame_count += 1

    finally:
        capture.release()
        writer.release()

    print(f"Frames processed: {frame_count}")
    print(f"Motion detections: {detection_count}")
    print(f"Feature records written: {csv_rows}")
    print(f"Annotated video: {destination}")
    print(f"Feature CSV: {csv_destination}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument(
        "--features-csv",
        default="results/motion_features.csv",
    )
    args = parser.parse_args()

    detect_motion_video(
        input_path=args.input,
        output_path=args.output,
        min_area=args.min_area,
        features_csv_path=args.features_csv,
    )

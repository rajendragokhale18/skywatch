"""End-to-end airborne-object detection and alert pipeline."""

import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2

from app.modules.alert_engine import AlertEngine
from app.modules.fusion import fuse_detection
from app.modules.motion_features import MotionFeatureExtractor
from app.modules.object_detector import ObjectDetector
from app.modules.tracker import CentroidTracker


class SkyWatchPipeline:
    def __init__(
        self,
        model_path="yolo26n.pt",
        confidence=0.25,
        image_size=640,
        tiled=True,
        alert_threshold=0.70,
        required_frames=5,
        cooldown_frames=30,
        device=None,
        allowed_classes=None,
    ):
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence=confidence,
            image_size=image_size,
            tile_size=image_size,
        )
        self.tiled = tiled
        self.device = device
        self.allowed_classes = {
            name.lower()
            for name in (
                allowed_classes
                if allowed_classes is not None
                else {
                    "airplane",
                    "bird",
                    "helicopter",
                    "kite",
                    "parachute",
                    "uav",
                }
            )
        }

        # Trackers are initialized after video FPS is known.
        self.tracker = None
        self.feature_extractor = None

        self.alert_engine = AlertEngine(
            threshold=alert_threshold,
            required_frames=required_frames,
            cooldown_frames=cooldown_frames,
        )

    @staticmethod
    def _xyxy_to_xywh(box):
        x1, y1, x2, y2 = map(int, box)
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def process_video(
        self,
        input_path,
        output_path,
        csv_path="results/skywatch_events.csv",
        features_path="results/skywatch_features.csv",
        class_diagnostics_path="results/skywatch_class_diagnostics.csv",
    ):
        source = Path(input_path)
        destination = Path(output_path)
        events_file = Path(csv_path)
        features_file = Path(features_path)
        diagnostics_file = Path(class_diagnostics_path)

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
        events_file.parent.mkdir(parents=True, exist_ok=True)
        features_file.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_file.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {destination}")

        self.tracker = CentroidTracker(
            max_disappeared=10,
            max_distance=max(60.0, min(width, height) * 0.05),
        )
        self.feature_extractor = MotionFeatureExtractor(
            history_size=30,
            fps=fps,
        )

        trails = defaultdict(lambda: deque(maxlen=30))
        frame_index = 0
        detection_count = 0
        alert_count = 0

        event_fields = [
            "frame", "timestamp_sec", "track_id", "class_name",
            "score", "reason", "x1", "y1", "x2", "y2",
        ]
        feature_fields = [
            "frame", "timestamp_sec", "track_id", "centroid_x",
            "centroid_y", "x1", "y1", "x2", "y2",
            "class_name", "confidence",
            "speed_px_per_sec", "straightness", "hover_ratio",
            "direction_change_deg", "observations",
        ]
        diagnostic_fields = [
            "frame", "timestamp_sec", "track_id",
            "raw_class", "stable_class", "class_changed", "confidence",
        ]

        try:
            with (
                events_file.open("w", newline="") as event_handle,
                features_file.open("w", newline="") as feature_handle,
                diagnostics_file.open("w", newline="") as diagnostic_handle,
            ):
                event_writer = csv.DictWriter(
                    event_handle, fieldnames=event_fields
                )
                feature_writer = csv.DictWriter(
                    feature_handle, fieldnames=feature_fields
                )
                diagnostic_writer = csv.DictWriter(
                    diagnostic_handle, fieldnames=diagnostic_fields
                )
                event_writer.writeheader()
                feature_writer.writeheader()
                diagnostic_writer.writeheader()

                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    detections = self.detector.detect(
                        frame, tiled=self.tiled
                    )
                    detections = [
                        detection
                        for detection in detections
                        if detection.get("class_name", "unknown").lower()
                        in self.allowed_classes
                    ]
                    detection_count += len(detections)

                    boxes = [
                        self._xyxy_to_xywh(item["bbox"])
                        for item in detections
                    ]
                    class_names = [
                        item["class_name"] for item in detections
                    ]
                    tracked = self.tracker.update(boxes, class_names)

                    track_results = {}
                    current_features = {}

                    for detection_index, track_id in (
                        self.tracker.assignments.items()
                    ):
                        detection = detections[detection_index]
                        x1, y1, x2, y2 = detection["bbox"]
                        tracked_center = tracked[track_id]

                        motion = self.feature_extractor.update(
                            track_id, tracked_center
                        )
                        current_features[track_id] = motion

                        # Use the track's smoothed class label to reduce
                        # frame-to-frame class flicker.
                        stable_class = detection["class_name"]
                        stable_detection = {
                            **detection,
                            "class_name": stable_class,
                        }

                        fused = fuse_detection(stable_detection, motion)
                        fused["track_id"] = track_id
                        track_results[track_id] = fused

                        diagnostic_writer.writerow({
                            "frame": frame_index,
                            "timestamp_sec": round(frame_index / fps, 3),
                            "track_id": track_id,
                            "raw_class": detection["class_name"],
                            "stable_class": stable_class,
                            "class_changed": (
                                detection["class_name"] != stable_class
                            ),
                            "confidence": detection["confidence"],
                        })

                        feature_writer.writerow({
                            "frame": frame_index,
                            "timestamp_sec": round(frame_index / fps, 3),
                            "track_id": track_id,
                            "centroid_x": tracked_center[0],
                            "centroid_y": tracked_center[1],
                            "x1": round(float(x1), 2),
                            "y1": round(float(y1), 2),
                            "x2": round(float(x2), 2),
                            "y2": round(float(y2), 2),
                            "class_name": stable_class,
                            "confidence": detection["confidence"],
                            **{
                                key: motion[key]
                                for key in (
                                    "speed_px_per_sec",
                                    "straightness",
                                    "hover_ratio",
                                    "direction_change_deg",
                                    "observations",
                                )
                            },
                        })

                        trails[track_id].append(tracked_center)

                        color = (0, 220, 0)
                        label = (
                            f"ID {track_id} {stable_class} "
                            f"{detection['confidence']:.2f}"
                        )
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            2,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (int(x1), max(20, int(y1) - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

                        points = list(trails[track_id])
                        for previous, current in zip(points, points[1:]):
                            cv2.line(
                                frame, previous, current, (255, 200, 0), 2
                            )

                    alerts = self.alert_engine.update(track_results)
                    alert_count += len(alerts)

                    for alert in alerts:
                        x1, y1, x2, y2 = alert["bbox"]
                        event_writer.writerow({
                            "frame": alert["frame_index"],
                            "timestamp_sec": round(
                                alert["frame_index"] / fps, 3
                            ),
                            "track_id": alert["track_id"],
                            "class_name": alert["class_name"],
                            "score": alert["score"],
                            "reason": alert["reason"],
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        })

                    cv2.putText(
                        frame,
                        f"Frame {frame_index} | Detections {len(detections)} "
                        f"| Alerts {alert_count}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    writer.write(frame)
                    frame_index += 1

        finally:
            capture.release()
            writer.release()

        summary = {
            "input": str(source),
            "output": str(destination),
            "frames_processed": frame_index,
            "detections": detection_count,
            "alerts": alert_count,
            "events_csv": str(events_file),
            "features_csv": str(features_file),
        }
        print(summary)
        return summary

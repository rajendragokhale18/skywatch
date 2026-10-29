
import math
from collections import defaultdict, deque

import numpy as np


class MotionFeatureExtractor:
    """Extract motion features from tracked object centroids."""

    def __init__(self, history_size: int = 30, fps: float = 20.0):
        if history_size < 2:
            raise ValueError("history_size must be at least 2.")
        if fps <= 0:
            raise ValueError("fps must be positive.")

        self.fps = fps
        self.history = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def update(self, object_id: int, centroid: tuple[int, int]) -> dict:
        points = self.history[object_id]
        points.append(centroid)

        coordinates = np.asarray(points, dtype=float)

        if len(coordinates) < 2:
            return {
                "object_id": object_id,
                "observations": len(coordinates),
                "speed_px_per_sec": 0.0,
                "straightness": 0.0,
                "hover_ratio": 0.0,
                "direction_change_deg": 0.0,
            }

        differences = np.diff(coordinates, axis=0)
        distances = np.linalg.norm(differences, axis=1)

        elapsed_seconds = (len(coordinates) - 1) / self.fps
        total_distance = float(distances.sum())

        speed = (
            total_distance / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        displacement = float(
            np.linalg.norm(coordinates[-1] - coordinates[0])
        )

        straightness = (
            displacement / total_distance
            if total_distance > 0
            else 0.0
        )

        # Fraction of frame-to-frame movements below 2 pixels.
        hover_ratio = float(np.mean(distances < 2.0))

        direction_change = 0.0
        if len(differences) >= 2:
            previous = differences[-2]
            current = differences[-1]

            previous_norm = np.linalg.norm(previous)
            current_norm = np.linalg.norm(current)

            if previous_norm > 0 and current_norm > 0:
                cosine = np.dot(previous, current) / (
                    previous_norm * current_norm
                )
                direction_change = math.degrees(
                    math.acos(float(np.clip(cosine, -1.0, 1.0)))
                )

        return {
            "object_id": object_id,
            "observations": len(coordinates),
            "speed_px_per_sec": round(speed, 2),
            "straightness": round(
                float(np.clip(straightness, 0.0, 1.0)), 3
            ),
            "hover_ratio": round(hover_ratio, 3),
            "direction_change_deg": round(direction_change, 2),
        }

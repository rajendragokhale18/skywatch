cat > app/modules/alert_engine.py <<'PY'
"""Generate alerts only after sustained qualifying detections."""

from collections import defaultdict
from typing import Any


class AlertEngine:
    def __init__(
        self,
        threshold: float = 0.70,
        required_frames: int = 5,
        cooldown_frames: int = 30,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1.")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames cannot be negative.")

        self.threshold = threshold
        self.required_frames = required_frames
        self.cooldown_frames = cooldown_frames
        self._streaks = defaultdict(int)
        self._last_alert_frame = {}
        self._frame_index = -1

    def update(
        self,
        tracked_scores: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Update alert state for one frame.

        tracked_scores maps track IDs to fusion results.
        Tracks absent from the current frame have their streak reset.
        """
        self._frame_index += 1
        alerts = []
        present_ids = set(tracked_scores)

        for track_id in list(self._streaks):
            if track_id not in present_ids:
                self._streaks[track_id] = 0

        for track_id, result in tracked_scores.items():
            score = float(result.get("fused_score", 0.0))

            if score < self.threshold:
                self._streaks[track_id] = 0
                continue

            self._streaks[track_id] += 1
            if self._streaks[track_id] < self.required_frames:
                continue

            last_alert = self._last_alert_frame.get(track_id)
            if (
                last_alert is not None
                and self._frame_index - last_alert < self.cooldown_frames
            ):
                continue

            alerts.append({
                "track_id": track_id,
                "frame_index": self._frame_index,
                "score": score,
                "class_name": result.get("class_name", "unknown"),
                "bbox": result.get("bbox"),
                "reason": "sustained_high_score",
            })
            self._last_alert_frame[track_id] = self._frame_index

        return alerts
PY
import math
from collections import Counter
from scipy.optimize import linear_sum_assignment

class CentroidTracker:
    """Track detections using bounding-box overlap, distance, and class history."""

    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: float = 60.0,
        iou_weight: float = 0.65,
    ):
        if max_disappeared < 0:
            raise ValueError("max_disappeared must be non-negative")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        if not 0 <= iou_weight <= 1:
            raise ValueError("iou_weight must be between 0 and 1")

        self.next_object_id = 0
        self.objects = {}
        self.boxes = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_weight = iou_weight

        self.classes = {}
        self.class_history = {}
        self.assignments = {}

    @staticmethod
    def _centroid(box):
        x, y, width, height = box
        return (x + width // 2, y + height // 2)

    @staticmethod
    def _xywh_to_xyxy(box):
        x, y, width, height = box
        return (x, y, x + width, y + height)

    @classmethod
    def _iou(cls, box_a, box_b):
        ax1, ay1, ax2, ay2 = cls._xywh_to_xyxy(box_a)
        bx1, by1, bx2, by2 = cls._xywh_to_xyxy(box_b)

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0
    @staticmethod
    def _linear_assignment(cost_matrix):
        """Return the minimum-cost one-to-one assignment."""
        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip(rows.tolist(), cols.tolist()))

    def update(self, boxes, class_names=None):
        """
        boxes: sequence of (x, y, width, height).
        class_names: optional class name for each box.

        Returns track ID -> centroid for active tracks.
        assignments maps detection index -> track ID for this frame.
        """
        if class_names is None:
            class_names = [None] * len(boxes)
        if len(class_names) != len(boxes):
            raise ValueError("class_names and boxes must have equal lengths")

        centroids = [self._centroid(box) for box in boxes]
        self.assignments = {}

        if not boxes:
            for track_id in list(self.objects):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
            return self.objects.copy()
        old_track_ids = list(self.objects.keys())
        detection_count = len(boxes)
        track_count = len(old_track_ids)

        # Build detection-to-track costs.
        invalid_cost = 1_000_000.0
        unmatched_detection_cost = 0.75
        unmatched_track_cost = 0.75

        size = detection_count + track_count
        cost_matrix = [
            [invalid_cost] * size
            for _ in range(size)
        ]

        # Real detection-to-track assignments.
        for detection_index, (box, centroid, class_name) in enumerate(
            zip(boxes, centroids, class_names)
        ):
            for track_index, track_id in enumerate(old_track_ids):
                old_centroid = self.objects[track_id]
                distance = math.dist(old_centroid, centroid)
                overlap = self._iou(self.boxes[track_id], box)

                # Reject distant matches unless box overlap is substantial.
                if distance > self.max_distance:
                    if overlap < 0.30:
                        continue


                distance_score = min(distance / self.max_distance, 1.0)
                cost = (
                    self.iou_weight * (1.0 - overlap)
                    + (1.0 - self.iou_weight) * distance_score
                )

                old_class = self.classes.get(track_id)
                if old_class and class_name and old_class != class_name:
                    cost += 0.20

                cost_matrix[detection_index][track_index] = cost

        # Dummy columns let detections remain unmatched.
        for detection_index in range(detection_count):
            for dummy_index in range(track_count, size):
                cost_matrix[detection_index][dummy_index] = (
                    unmatched_detection_cost
                )

        # Dummy rows let existing tracks remain unmatched.
        for dummy_index in range(detection_count, size):
            for track_index in range(track_count):
                cost_matrix[dummy_index][track_index] = unmatched_track_cost

        # Dummy-to-dummy assignments have no cost.
        for dummy_index in range(detection_count, size):
            for dummy_column in range(track_count, size):
                cost_matrix[dummy_index][dummy_column] = 0.0

        assignments = self._linear_assignment(cost_matrix)
        used_detections = set()
        used_tracks = set()

        for detection_index, track_index in assignments:
            if (
                detection_index >= detection_count
                or track_index >= track_count
            ):
                continue

            if cost_matrix[detection_index][track_index] >= invalid_cost:
                continue

            track_id = old_track_ids[track_index]
            self.objects[track_id] = centroids[detection_index]
            self.boxes[track_id] = boxes[detection_index]
            self.disappeared[track_id] = 0
            self.assignments[detection_index] = track_id
            self._record_class(track_id, class_names[detection_index])

            used_detections.add(detection_index)
            used_tracks.add(track_id)

        for track_id in list(self.objects):
            if track_id not in used_tracks:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)

        for detection_index, (box, centroid) in enumerate(zip(boxes, centroids)):
            if detection_index in used_detections:
                continue
            track_id = self._register(box, centroid, class_names[detection_index])
            self.assignments[detection_index] = track_id

        return self.objects.copy()

    def _record_class(self, track_id, class_name):
        if not class_name:
            return

        history = self.class_history.setdefault(track_id, [])
        history.append(class_name)
        del history[:-15]

        current_class = self.classes.get(track_id)

        # Establish the initial class from the first valid detection.
        if current_class is None:
            self.classes[track_id] = class_name
            return

        # Keep the established class unless another class has sustained
        # support in the recent observation window.
        recent_counts = Counter(history)
        candidate_class, candidate_count = recent_counts.most_common(1)[0]

        if candidate_class == current_class:
            return

        required_support = 8
        if (
            candidate_count >= required_support
            and candidate_count > recent_counts.get(current_class, 0)
        ):
            self.classes[track_id] = candidate_class

    def _register(self, box, centroid, class_name=None):
        track_id = self.next_object_id
        self.next_object_id += 1
        self.objects[track_id] = centroid
        self.boxes[track_id] = box
        self.disappeared[track_id] = 0
        self.class_history[track_id] = []
        self._record_class(track_id, class_name)
        return track_id

    def _deregister(self, track_id):
        self.objects.pop(track_id, None)
        self.boxes.pop(track_id, None)
        self.disappeared.pop(track_id, None)
        self.classes.pop(track_id, None)
        self.class_history.pop(track_id, None)

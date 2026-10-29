
from app.modules.tracker import CentroidTracker


def test_tracker_preserves_id_when_object_moves():
    tracker = CentroidTracker(max_distance=100)

    first = tracker.update([(10, 10, 20, 20)])
    second = tracker.update([(15, 10, 20, 20)])

    assert list(first.keys()) == [0]
    assert list(second.keys()) == [0]
    assert second[0] == (25, 20)


def test_tracker_registers_new_object():
    tracker = CentroidTracker()

    tracker.update([(10, 10, 20, 20)])
    result = tracker.update([
        (12, 10, 20, 20),
        (300, 300, 20, 20),
    ])

    assert len(result) == 2


def test_tracker_removes_disappeared_object():
    tracker = CentroidTracker(max_disappeared=1)

    tracker.update([(10, 10, 20, 20)])
    tracker.update([])
    tracker.update([])

    assert tracker.objects == {}



def test_tracker_keeps_id_when_class_temporarily_changes():
    tracker = CentroidTracker(max_distance=100)

    tracker.update([(10, 10, 20, 20)], ["uav"])
    result = tracker.update([(12, 10, 20, 20)], ["bird"])

    assert list(result.keys()) == [0]
    assert tracker.assignments == {0: 0}
    # Majority history should preserve the established class.
    assert tracker.classes[0] == "uav"


def test_tracker_assigns_each_detection_to_a_distinct_track():
    tracker = CentroidTracker(max_distance=100)

    tracker.update(
        [(10, 10, 20, 20), (80, 10, 20, 20)],
        ["bird", "bird"],
    )
    tracker.update(
        [(14, 10, 20, 20), (76, 10, 20, 20)],
        ["bird", "bird"],
    )

    assert len(tracker.assignments) == 2
    assert len(set(tracker.assignments.values())) == 2
    assert tracker.assignments == {0: 0, 1: 1}


def test_tracker_avoids_greedy_assignment_when_global_match_is_better():
    tracker = CentroidTracker(max_distance=100, iou_weight=0.5)

    # Initial detections establish track 0 and track 1.
    tracker.update([
        (0, 0, 20, 20),
        (30, 0, 20, 20),
    ], ["bird", "bird"])

    # The next detections are positioned so that the globally best
    # one-to-one assignment should preserve both track identities.
    result = tracker.update([
        (12, 0, 20, 20),
        (0, 0, 20, 20),
    ], ["bird", "bird"])

    assert tracker.assignments == {0: 1, 1: 0}
    assert len(result) == 2
def test_linear_assignment_finds_global_minimum():
    tracker = CentroidTracker()

    cost_matrix = [
        [1.0, 2.0],
        [2.0, 100.0],
    ]

    assignments = tracker._linear_assignment(cost_matrix)

    assert assignments == [(0, 1), (1, 0)]
    assert sum(cost_matrix[row][col] for row, col in assignments) == 4.0


def test_tracker_does_not_match_distant_object_with_tiny_overlap():
    tracker = CentroidTracker(max_distance=60)

    tracker.update([(0, 0, 100, 100)], ["bird"])

    # The boxes overlap by only a narrow strip, but their centroids
    # are too far apart to represent a plausible match.
    tracker.update([(99, 0, 100, 100)], ["bird"])

    assert len(tracker.assignments) == 1
    assert tracker.assignments[0] == 1
    assert 0 in tracker.objects
    assert 1 in tracker.objects


def test_tracker_preserves_id_for_reasonable_motion_with_class_change():
    tracker = CentroidTracker(max_distance=100)

    tracker.update([(10, 10, 20, 20)], ["uav"])
    tracker.update([(18, 10, 20, 20)], ["bird"])

    assert tracker.assignments == {0: 0}
    assert len(tracker.objects) == 1


def test_tracker_class_stays_stable_during_repeated_temporary_noise():
    tracker = CentroidTracker(max_distance=100)

    tracker.update([(10, 10, 20, 20)], ["uav"])

    for _ in range(4):
        tracker.update([(12, 10, 20, 20)], ["bird"])

    assert tracker.classes[0] == "uav"


def test_tracker_class_can_change_after_sustained_new_evidence():
    tracker = CentroidTracker(max_distance=100)

    tracker.update([(10, 10, 20, 20)], ["uav"])

    for _ in range(20):
        tracker.update([(12, 10, 20, 20)], ["bird"])

    assert tracker.classes[0] == "bird"

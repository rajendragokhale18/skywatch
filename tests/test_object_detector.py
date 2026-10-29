from app.modules.object_detector import ObjectDetector


def make_detection(class_id, class_name, confidence, bbox):
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
    }


def test_class_agnostic_nms_suppresses_near_identical_boxes():
    detections = [
        make_detection(0, "airplane", 0.80, [100, 100, 200, 200]),
        make_detection(1, "bird", 0.70, [102, 102, 198, 198]),
    ]

    kept = ObjectDetector._nms(
        detections, iou_threshold=0.90, class_agnostic=True
    )

    assert len(kept) == 1
    assert kept[0]["class_name"] == "airplane"


def test_class_agnostic_nms_preserves_boxes_below_threshold():
    detections = [
        make_detection(0, "airplane", 0.80, [100, 100, 200, 200]),
        make_detection(1, "bird", 0.70, [130, 100, 230, 200]),
    ]

    kept = ObjectDetector._nms(
        detections, iou_threshold=0.90, class_agnostic=True
    )

    assert len(kept) == 2


def test_default_nms_preserves_different_classes():
    detections = [
        make_detection(0, "airplane", 0.80, [100, 100, 200, 200]),
        make_detection(1, "bird", 0.70, [102, 102, 198, 198]),
    ]

    kept = ObjectDetector._nms(detections, iou_threshold=0.90)

    assert len(kept) == 2


from app.modules.motion_features import MotionFeatureExtractor


def test_stationary_object_has_high_hover_ratio():
    extractor = MotionFeatureExtractor(fps=20)

    extractor.update(1, (100, 100))
    features = extractor.update(1, (100, 100))
    features = extractor.update(1, (100, 100))

    assert features["hover_ratio"] == 1.0
    assert features["speed_px_per_sec"] == 0.0


def test_straight_motion_has_high_straightness():
    extractor = MotionFeatureExtractor(fps=10)

    extractor.update(1, (0, 0))
    extractor.update(1, (10, 0))
    features = extractor.update(1, (20, 0))

    assert features["straightness"] == 1.0
    assert features["speed_px_per_sec"] == 100.0


def test_invalid_fps_is_rejected():
    try:
        MotionFeatureExtractor(fps=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

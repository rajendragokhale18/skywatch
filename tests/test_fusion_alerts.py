import pytest

from app.modules.fusion import fuse_detection
from app.modules.alert_engine import AlertEngine


def test_fusion_score_is_bounded():
    result = fuse_detection(
        {"class_name": "uav", "confidence": 0.9},
        {"observations": 5, "straightness": 0.8, "hover_ratio": 0.2},
    )
    assert 0.0 <= result["fused_score"] <= 1.0
    assert result["class_name"] == "uav"


def test_fusion_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        fuse_detection({"confidence": 1.5})


def test_alert_requires_sustained_score():
    engine = AlertEngine(threshold=0.7, required_frames=3)
    item = {"fused_score": 0.9, "class_name": "uav"}
    assert engine.update({1: item}) == []
    assert engine.update({1: item}) == []
    assert len(engine.update({1: item})) == 1


def test_alert_resets_after_low_score():
    engine = AlertEngine(threshold=0.7, required_frames=3)
    high = {"fused_score": 0.9}
    low = {"fused_score": 0.2}
    engine.update({1: high})
    engine.update({1: high})
    engine.update({1: low})
    assert engine.update({1: high}) == []


def test_alert_cooldown_prevents_immediate_repeat():
    engine = AlertEngine(
        threshold=0.7,
        required_frames=1,
        cooldown_frames=3,
    )
    item = {"fused_score": 0.9}
    assert len(engine.update({1: item})) == 1
    assert engine.update({1: item}) == []
    assert engine.update({1: item}) == []
    assert len(engine.update({1: item})) == 1

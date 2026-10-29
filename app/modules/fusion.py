"""Combine detector confidence with motion evidence."""


def fuse_detection(
    detection: dict,
    motion: dict | None = None,
    appearance_weight: float = 0.65,
) -> dict:
    if not 0.0 <= appearance_weight <= 1.0:
        raise ValueError("appearance_weight must be between 0 and 1.")

    confidence = float(detection.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    motion = motion or {}
    straightness = float(motion.get("straightness", 0.0))
    hover_ratio = float(motion.get("hover_ratio", 0.0))
    observations = int(motion.get("observations", 0))

    if not 0.0 <= straightness <= 1.0:
        raise ValueError("straightness must be between 0 and 1.")
    if not 0.0 <= hover_ratio <= 1.0:
        raise ValueError("hover_ratio must be between 0 and 1.")

    motion_score = (
        0.5
        if observations < 3
        else 0.5 * straightness + 0.5 * hover_ratio
    )

    fused_score = (
        appearance_weight * confidence
        + (1.0 - appearance_weight) * motion_score
    )

    return {
        "class_name": detection.get("class_name", "unknown"),
        "class_id": detection.get("class_id"),
        "bbox": detection.get("bbox"),
        "appearance_confidence": confidence,
        "motion_score": round(motion_score, 4),
        "fused_score": round(fused_score, 4),
        "motion_observations": observations,
        "evidence": {
            "straightness": straightness,
            "hover_ratio": hover_ratio,
        },
    }

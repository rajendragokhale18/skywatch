"""Central configuration for SkyWatch."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv("SKYWATCH_MODEL", str(PROJECT_ROOT / "yolo26n.pt"))
)

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results" / "api"

CONFIDENCE_THRESHOLD = float(
    os.getenv("SKYWATCH_CONFIDENCE", "0.25")
)
IMAGE_SIZE = int(os.getenv("SKYWATCH_IMAGE_SIZE", "640"))
ALERT_THRESHOLD = float(os.getenv("SKYWATCH_ALERT_THRESHOLD", "0.70"))

MAX_UPLOAD_BYTES = int(
    os.getenv("SKYWATCH_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024))
)

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

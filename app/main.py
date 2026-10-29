"""FastAPI application for SkyWatch video processing."""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import (
    ALLOWED_VIDEO_SUFFIXES,
    ALERT_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
    RESULTS_DIR,
    UPLOAD_DIR,
)
from app.pipeline import SkyWatchPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.startup_error = None

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if not MODEL_PATH.is_file() and str(MODEL_PATH) != "yolo26n.pt":
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        app.state.pipeline = SkyWatchPipeline(
            model_path=str(MODEL_PATH),
            confidence=CONFIDENCE_THRESHOLD,
            image_size=IMAGE_SIZE,
            tiled=False,
            alert_threshold=ALERT_THRESHOLD,
        )
    except Exception as exc:
        app.state.startup_error = str(exc)

    yield


app = FastAPI(
    title="SkyWatch API",
    description="Airborne-object detection and video analysis.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "SkyWatch",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if app.state.pipeline is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "detail": app.state.startup_error,
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    }


@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    if app.state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline unavailable: {app.state.startup_error}",
        )

    filename = Path(file.filename or "upload.mp4")
    suffix = filename.suffix.lower()

    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video extension: {suffix or '(none)'}",
        )

    job_id = uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = RESULTS_DIR / f"{job_id}_annotated.mp4"
    events_path = RESULTS_DIR / f"{job_id}_events.csv"
    features_path = RESULTS_DIR / f"{job_id}_features.csv"

    size = 0
    try:
        with input_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the size limit.",
                    )
                destination.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = await run_in_threadpool(
            app.state.pipeline.process_video,
            str(input_path),
            str(output_path),
            str(events_path),
            str(features_path),
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "summary": result,
            "artifacts": {
                "annotated_video": str(output_path),
                "events_csv": str(events_path),
                "features_csv": str(features_path),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {exc}",
        ) from exc
    finally:
        await file.close()
        input_path.unlink(missing_ok=True)


@app.get("/jobs/{job_id}/artifacts/{artifact_type}")
def download_artifact(job_id: str, artifact_type: str):
    """Download an output artifact using its job ID."""

    if len(job_id) != 32 or any(
        char not in "0123456789abcdef" for char in job_id
    ):
        raise HTTPException(status_code=400, detail="Invalid job ID.")

    artifact_suffixes = {
        "video": ("_annotated.mp4", "video/mp4"),
        "events": ("_events.csv", "text/csv"),
        "features": ("_features.csv", "text/csv"),
    }

    if artifact_type not in artifact_suffixes:
        raise HTTPException(
            status_code=404,
            detail="Unknown artifact type.",
        )

    suffix, media_type = artifact_suffixes[artifact_type]
    artifact_path = (RESULTS_DIR / f"{job_id}{suffix}").resolve()

    if artifact_path.parent != RESULTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid artifact path.")

    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    from fastapi.responses import FileResponse

    return FileResponse(
        path=artifact_path,
        media_type=media_type,
        filename=artifact_path.name,
    )

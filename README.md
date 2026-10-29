# SkyWatch

**Airborne Object Detection, Tracking, and Video Alert Pipeline**

SkyWatch is a computer-vision project that analyzes video for airborne objects, associates detections across frames, extracts motion features, and generates alerts based on sustained detection evidence.

## Overview

The system combines object detection, multi-object tracking, motion analysis, score fusion, and an HTTP API into an end-to-end video-processing pipeline.

### Supported classes

- Airplane
- Bird
- Helicopter
- Kite
- Parachute
- UAV (drone)

### Key capabilities

- Detect airborne objects in uploaded video.
- Associate detections across frames using bounding-box overlap, centroid distance, and assignment optimization.
- Extract motion features, including speed, straightness, hover ratio, and direction changes.
- Combine detection confidence and motion evidence into a heuristic score.
- Generate alerts when a track satisfies configurable score and persistence conditions.
- Export an annotated MP4, event CSV, and feature CSV.
- Process video through a FastAPI service.

## Architecture

```text
Input video
    |
    v
Video decoding (OpenCV)
    |
    v
Airborne-object detector
    |
    v
Detection filtering
    |
    v
Centroid tracking and class stabilization
    |
    v
Motion feature extraction
    |
    v
Detection/motion score fusion
    |
    v
Sustained-score alert engine
    |
    +--> Annotated video
    +--> Events CSV
    +--> Features CSV
```

## Technology stack

- Python 3.12
- PyTorch and Ultralytics
- OpenCV
- SciPy
- FastAPI and Uvicorn
- Pytest
- Docker (container configuration included)

## Repository structure

```text
skywatch/
├── app/
│   ├── main.py
│   ├── pipeline.py
│   ├── config.py
│   └── modules/
│       ├── object_detector.py
│       ├── tracker.py
│       ├── motion_features.py
│       ├── fusion.py
│       ├── alert_engine.py
│       ├── media_io.py
│       └── motion_candidates.py
├── data/
│   ├── airborne_dataset/
│   └── videos/
├── runs/
│   └── detect/results/training/mbaod_baseline/
├── results/
├── scripts/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

Python 3.12 is recommended.

Create and activate a Conda environment:

```bash
conda create -n skywatch python=3.12 -y
conda activate skywatch
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Select the trained checkpoint

The repository contains a trained checkpoint at:

```text
runs/detect/results/training/mbaod_baseline/weights/best.pt
```

Configure the API to use it:

```bash
export SKYWATCH_MODEL="$PWD/runs/detect/results/training/mbaod_baseline/weights/best.pt"
export SKYWATCH_IMAGE_SIZE=1280
```

The image-size setting above matches the previously used local API configuration. Higher image sizes can increase inference time and memory use.

The application configuration may default to `yolo26n.pt` if `SKYWATCH_MODEL` is not set. That file is the base model checkpoint, not the trained SkyWatch checkpoint.

## Run the API

Start the server:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Check health:

```bash
curl -s http://127.0.0.1:8000/health
```

A healthy response reports `"status": "ok"` and `"model_loaded": true`.

## Process a video

The API accepts video uploads at `POST /process-video`.

```bash
curl -s -X POST \
  -F "file=@data/videos/drone_test.mp4" \
  http://127.0.0.1:8000/process-video
```

The response contains a job ID, processing summary, and artifact paths.

Supported video extensions are configured by the application. The current API configuration allows MP4, AVI, MOV, and MKV files and enforces an upload-size limit.

### Download artifacts

Replace `<JOB_ID>` with the 32-character job ID returned by the API.

Download the annotated video:

```bash
curl -L \
  http://127.0.0.1:8000/jobs/<JOB_ID>/artifacts/video \
  -o annotated_video.mp4
```

Download alert events:

```bash
curl -L \
  http://127.0.0.1:8000/jobs/<JOB_ID>/artifacts/events \
  -o events.csv
```

Download per-detection features:

```bash
curl -L \
  http://127.0.0.1:8000/jobs/<JOB_ID>/artifacts/features \
  -o features.csv
```

## Model and evaluation

The trained checkpoint was produced using an airborne-object dataset configured through:

```text
data/airborne_dataset/mbaod_yolo.yaml
```

The training configuration specifies a YOLO26 nano model, 640-pixel training images, batch size 4, and 30 configured epochs. However, the available training history records only four epochs. The reason training stopped is not established by the available log.

### Recorded validation metrics

The following values are from the fourth recorded epoch in `results.csv`:

| Metric | Recorded value |
|---|---:|
| Precision | 0.8466 |
| Recall | 0.7285 |
| mAP@50 | 0.8137 |
| mAP@50–95 | 0.5216 |

These are validation metrics from the recorded training run. They are not an independent test-set result, and they do not establish production accuracy.

Previously recorded clean-test evaluation results were approximately:

| Metric | Recorded value |
|---|---:|
| Precision | 0.8443 |
| Recall | 0.7461 |
| mAP@50 | 0.8232 |
| mAP@50–95 | 0.5301 |

The evaluation script and dataset split should be checked before presenting the second table as an independently reproducible benchmark.

### Known limitations

- UAV recall was previously measured at approximately 0.57 in class-wise evaluation, indicating that UAV detection needs further improvement.
- Small, distant, or visually ambiguous objects may be missed or misclassified.
- Tracking associates detections; it does not guarantee persistent identity in every frame.
- The fused score is a heuristic, not a calibrated probability of threat.
- Alerts are generated by configurable score and persistence rules. They should not be interpreted as confirmed security threats.
- Performance depends on video resolution, lighting, object size, camera motion, and inference settings.

## Tests

Run the test suite:

```bash
python -m pytest -q
```

The latest recorded run completed with 30 passing tests and one Starlette deprecation warning.

## Configuration

The application reads its principal settings from `app/config.py`, including:

- `SKYWATCH_MODEL` — model checkpoint path
- `SKYWATCH_IMAGE_SIZE` — inference image size
- `SKYWATCH_CONFIDENCE` — detection confidence threshold
- `SKYWATCH_ALERT_THRESHOLD` — fused-score threshold for alerting
- `SKYWATCH_MAX_UPLOAD_BYTES` — maximum accepted upload size

Check `app/config.py` for the exact defaults and supported configuration.

## Future improvements

1. Complete training and evaluate the best checkpoint on a documented, independent test split.
2. Improve UAV recall and measure class-wise precision and recall.
3. Benchmark inference latency and throughput on representative hardware.
4. Evaluate tracking identity consistency and class stabilization quantitatively.
5. Add a polished monitoring interface and structured downloadable reports.
6. Validate container builds and document deployment configuration.

## Responsible use

SkyWatch is a computer-vision demonstration and research project. Detection results and heuristic alerts require human review and should not be treated as definitive identification or threat assessment.

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "SkyWatch"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_rejects_unsupported_file_type():
    with TestClient(app) as client:
        response = client.post(
            "/process-video",
            files={
                "file": (
                    "notes.txt",
                    b"not a video",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 400
    assert "Unsupported video extension" in response.json()["detail"]


def test_rejects_empty_video_upload():
    with TestClient(app) as client:
        response = client.post(
            "/process-video",
            files={
                "file": (
                    "empty.mp4",
                    b"",
                    "video/mp4",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_download_artifact_success(tmp_path, monkeypatch):
    import app.main as main_module

    job_id = "a" * 32
    artifact = tmp_path / f"{job_id}_annotated.mp4"
    artifact.write_bytes(b"fake-video-content")

    monkeypatch.setattr(main_module, "RESULTS_DIR", tmp_path)

    with TestClient(app) as client:
        response = client.get(
            f"/jobs/{job_id}/artifacts/video"
        )

    assert response.status_code == 200
    assert response.content == b"fake-video-content"
    assert response.headers["content-type"] == "video/mp4"


def test_download_artifact_not_found(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "RESULTS_DIR", tmp_path)

    with TestClient(app) as client:
        response = client.get(
            f"/jobs/{'b' * 32}/artifacts/video"
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Artifact not found."


def test_download_artifact_rejects_invalid_job_id():
    with TestClient(app) as client:
        response = client.get(
            "/jobs/invalid/artifacts/video"
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid job ID."


def test_download_artifact_rejects_unknown_type():
    with TestClient(app) as client:
        response = client.get(
            f"/jobs/{'c' * 32}/artifacts/unknown"
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown artifact type."

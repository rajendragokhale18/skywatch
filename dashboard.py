from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="SkyWatch | Video Intelligence",
    page_icon="S",
    layout="wide",
)

API = "http://127.0.0.1:8000"
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.8rem;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,.08);
    padding: 14px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.18);
}
</style>
""", unsafe_allow_html=True)

st.title("SKYWATCH")
st.caption("AIRBORNE OBJECT DETECTION  /  TRACKING  /  VIDEO ANALYTICS")
st.divider()

with st.sidebar:
    st.subheader("System status")
    try:
        health_response = requests.get(f"{API}/health", timeout=5)
        health_response.raise_for_status()
        health = health_response.json()

        if health.get("model_loaded"):
            st.success("Detection model online")
            st.caption(
                f"Checkpoint: {Path(health.get('model_path', 'unknown')).name}"
            )
        else:
            st.error("Model not loaded")
            st.json(health)
    except requests.RequestException:
        st.error("API unavailable")
        st.caption("Start the FastAPI service on port 8000.")

    st.divider()
    st.caption("Formats: MP4 · AVI · MOV · MKV")
    st.caption("Analytics describe model outputs, not ground-truth counts.")

upload_col, output_col = st.columns([0.9, 1.1], gap="large")

with upload_col:
    st.subheader("01 / Input video")
    uploaded = st.file_uploader("Upload video", type=VIDEO_TYPES)

    if uploaded:
        st.video(uploaded)
        st.caption(
            f"{uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MB"
        )

    process = st.button(
        "Analyze video",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

if process and uploaded:
    try:
        with st.spinner(
            "Processing video: detecting objects, tracking motion, and "
            "generating artifacts..."
        ):
            response = requests.post(
                f"{API}/process-video",
                files={
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                },
                timeout=(10, 1800),
            )

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"Processing failed ({response.status_code}): {detail}")
        else:
            st.session_state["skywatch_result"] = response.json()
            st.session_state["skywatch_job_id"] = response.json().get("job_id")

    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")

result = st.session_state.get("skywatch_result")

with output_col:
    st.subheader("02 / Annotated output")

    if not result:
        st.info("Upload a video and select Analyze video to begin.")
    else:
        job_id = result.get("job_id")
        if not job_id:
            st.error("API response did not include a job ID.")
        else:
            try:
                video_response = requests.get(
                    f"{API}/jobs/{job_id}/artifacts/video",
                    timeout=120,
                )
                video_response.raise_for_status()
                video_bytes = video_response.content

                st.video(video_bytes)
                st.success("Video processing completed")
                st.download_button(
                    "Download annotated MP4",
                    data=video_bytes,
                    file_name=f"skywatch_{job_id}_annotated.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key=f"{job_id}_video",
                )
            except requests.RequestException as exc:
                st.error(f"Could not retrieve annotated video: {exc}")

            summary = result.get("summary") or {}
            st.markdown("#### Processing summary")

            metrics = [
                ("frames_processed", "Frames processed"),
                ("detections", "Detection observations"),
                ("alerts", "Alerts"),
            ]
            metric_cols = st.columns(3)
            for col, (key, label) in zip(metric_cols, metrics):
                col.metric(label, summary.get(key, "—"))

            st.caption(
                "Detection observations count detections across frames; "
                "they are not unique objects."
            )

            csv_data = {}
            for kind in ("features", "events"):
                try:
                    csv_response = requests.get(
                        f"{API}/jobs/{job_id}/artifacts/{kind}",
                        timeout=60,
                    )
                    csv_response.raise_for_status()
                    csv_data[kind] = csv_response.content
                except requests.RequestException:
                    csv_data[kind] = None

            st.markdown("#### Export artifacts")
            download_cols = st.columns(2)
            for col, kind, label in [
                (download_cols[0], "features", "Download features CSV"),
                (download_cols[1], "events", "Download events CSV"),
            ]:
                if csv_data[kind] is not None:
                    col.download_button(
                        label,
                        data=csv_data[kind],
                        file_name=f"skywatch_{job_id}_{kind}.csv",
                        mime="text/csv",
                        key=f"{job_id}_{kind}",
                        use_container_width=True,
                    )
                else:
                    col.warning(f"{kind.title()} CSV unavailable.")

            if csv_data.get("features"):
                try:
                    from io import BytesIO

                    features = pd.read_csv(BytesIO(csv_data["features"]))
                except (ValueError, pd.errors.ParserError):
                    features = pd.DataFrame()
            else:
                features = pd.DataFrame()

            if csv_data.get("events"):
                try:
                    events = pd.read_csv(BytesIO(csv_data["events"]))
                except (ValueError, pd.errors.ParserError):
                    events = pd.DataFrame()
            else:
                events = pd.DataFrame()

            st.divider()
            st.subheader("03 / Detection analytics")

            if features.empty:
                st.warning("No feature observations were returned for this job.")
            else:
                required = {"class_name", "track_id", "confidence"}
                if not required.issubset(features.columns):
                    st.warning("The feature CSV schema is incomplete.")
                else:
                    class_counts = features["class_name"].fillna(
                        "unknown"
                    ).value_counts()

                    unique_tracks = features["track_id"].nunique()
                    mean_confidence = features["confidence"].mean()

                    a, b, c = st.columns(3)
                    a.metric("Unique track IDs", int(unique_tracks))
                    b.metric(
                        "Mean detection confidence",
                        f"{mean_confidence:.1%}"
                        if pd.notna(mean_confidence) else "—",
                    )
                    c.metric("Classes observed", int(class_counts.size))

                    chart_col, confidence_col = st.columns(2, gap="large")
                    with chart_col:
                        st.markdown("##### Class-wise observations")
                        st.bar_chart(
                            class_counts.rename("observations"),
                            y="observations",
                        )
                        st.caption(
                            "Counts are frame-level observations, not "
                            "unique objects or ground-truth labels."
                        )

                    with confidence_col:
                        st.markdown("##### Confidence distribution")
                        st.bar_chart(
                            features["confidence"]
                            .dropna()
                            .clip(0, 1)
                            .value_counts(
                                bins=10, sort=False
                            )
                            .sort_index()
                            .rename("observations")
                        )

                    st.markdown("##### Track summary")
                    aggregations = {
                        "observations": ("track_id", "size"),
                        "mean_confidence": ("confidence", "mean"),
                    }
                    if "speed_px_per_sec" in features.columns:
                        aggregations["max_speed_px_per_sec"] = (
                            "speed_px_per_sec", "max"
                        )

                    track_summary = (
                        features.groupby(["track_id", "class_name"], dropna=False)
                        .agg(**aggregations)
                        .sort_values("observations", ascending=False)
                        .reset_index()
                    )
                    st.dataframe(
                        track_summary.round(3),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.divider()
            st.subheader("04 / Alert events")
            if events.empty:
                st.info(
                    "No alert events were triggered for this video. "
                    "Objects may still have been detected and tracked."
                )
            else:
                st.metric("Alert events", len(events))
                st.dataframe(events, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "SKYWATCH · Local video inference · Model predictions can be incorrect "
    "and should be validated before operational use."
)

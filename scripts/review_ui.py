"""Local test page for the moderation API. Dev tool only: app/ never imports this.

Streamlit needs pillow<13 but the service uses Pillow 13, so the page runs in its
own isolated environment. Start it from the project folder (API server running):

    uvx --with httpx --with pandas --with pillow-heif streamlit run scripts/review_ui.py

"Local details" calls the project's own code through `uv run`, never by import.
"""
import json
import os
import subprocess
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener

PROJECT_ROOT = Path(__file__).resolve().parents[1]

register_heif_opener()

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".avi"}
DECISION_COLOR = {"APPROVED": "green", "HUMAN_REVIEW": "orange", "REJECTED": "red"}


def guess_type(name: str) -> str:
    return "video" if Path(name).suffix.lower() in VIDEO_EXT else "image"


def moderate(api_url, name, data, content_type, user_id, context):
    start = time.perf_counter()
    response = httpx.post(
        api_url,
        files={"file": (name, data)},
        data={"content_type": content_type, "user_id": user_id, "context": context},
        timeout=180,
    )
    round_trip_ms = round((time.perf_counter() - start) * 1000)
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    return response.status_code, body, round_trip_ms


def to_row(name, content_type, status, body, round_trip_ms):
    flags = ", ".join(body.get("flags", [])) if "flags" in body else str(body.get("detail"))
    return {
        "file": name,
        "type": content_type,
        "http": status,
        "decision": body.get("decision"),
        "code": body.get("rejection_code"),
        "flags": flags,
        "server_ms": body.get("processing_ms"),
        "round_trip_ms": round_trip_ms,
        "content_id": (body.get("content_id") or "")[:8],
    }


def inspect_locally(name, data, content_type):
    """Run the project's validation in its own environment to show metadata and pHash."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # use the project's .venv
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as folder:
        path = Path(folder) / name
        path.write_bytes(data)
        result = subprocess.run(
            ["uv", "run", "python", "-m", "scripts.inspect_file", str(path), content_type, "--json"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        return {"error": result.stderr.strip()[-2000:]}
    return json.loads(result.stdout.strip().splitlines()[-1])


def preview(data, content_type):
    if content_type == "video":
        st.video(data)
        return
    try:
        with Image.open(BytesIO(data)) as img:
            img.thumbnail((480, 480))
            st.image(img.convert("RGB"))
    except Exception:
        st.caption("No preview (the file cannot be decoded).")


# --- page --------------------------------------------------------------------

st.set_page_config(page_title="MMP Moderation Tester", layout="wide")
st.title("MeetMyPets moderation tester")

with st.sidebar:
    api_url = st.text_input("API URL", "http://localhost:8000/api/v1/moderate")
    user_id = st.text_input("user_id", "3f2b8c1e-0000-4000-8000-000000000001")
    context = st.selectbox("context", ["post", "profile", "breeding", "story"])
    type_mode = st.radio("content_type", ["auto (from file extension)", "image", "video"])
    show_local = st.checkbox("Show local details (metadata, pHash)")
    if st.button("Check server"):
        try:
            health = httpx.get(api_url.split("/api/")[0] + "/health", timeout=5).json()
            st.success(f"Server says: {health}")
        except Exception as exc:
            st.error(f"Server not reachable: {exc}")


def content_type_for(name: str) -> str:
    return guess_type(name) if type_mode.startswith("auto") else type_mode


upload_tab, folder_tab = st.tabs(["Upload files", "Run a folder (timings)"])

with upload_tab:
    files = st.file_uploader("Drop images or videos", accept_multiple_files=True)
    if files and st.button("Moderate", type="primary"):
        rows = []
        for f in files:
            data = f.getvalue()
            content_type = content_type_for(f.name)
            status, body, ms = moderate(api_url, f.name, data, content_type, user_id, context)
            rows.append(to_row(f.name, content_type, status, body, ms))

            with st.container(border=True):
                left, right = st.columns([1, 2])
                with left:
                    preview(data, content_type)
                with right:
                    decision = body.get("decision") or f"HTTP {status}"
                    color = DECISION_COLOR.get(decision, "gray")
                    st.markdown(f"### {f.name}  :{color}[{decision}]")
                    st.write(
                        f"**code:** {body.get('rejection_code')}  ·  "
                        f"**flags:** {rows[-1]['flags']}  ·  "
                        f"**server:** {body.get('processing_ms')} ms  ·  **round trip:** {ms} ms"
                    )
                    with st.expander("Full JSON response"):
                        st.json(body)
                    if show_local:
                        with st.expander("Local details (metadata, pHash)"):
                            st.json(inspect_locally(f.name, data, content_type))

        st.subheader("Summary")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with folder_tab:
    folder = st.text_input("Folder on this PC", "D:/samples")
    runs = st.number_input("Runs per file", min_value=1, max_value=10, value=3)
    bypass_cache = st.checkbox(
        "Bypass the cache (new random user_id per run, so every run is really processed)",
        value=True,
    )
    if st.button("Run folder", type="primary"):
        folder_path = Path(folder)
        if not folder_path.is_dir():
            st.error(f"Not a folder: {folder}")
            st.stop()
        paths = sorted(p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
        rows = []
        progress = st.progress(0.0)
        total = len(paths) * runs
        for i, path in enumerate(paths):
            data = path.read_bytes()
            content_type = content_type_for(path.name)
            for run in range(runs):
                uid = str(uuid.uuid4()) if bypass_cache else user_id
                status, body, ms = moderate(api_url, path.name, data, content_type, uid, context)
                rows.append(to_row(path.name, content_type, status, body, ms) | {"run": run + 1})
                progress.progress((i * runs + run + 1) / max(total, 1))

        if not rows:
            st.warning("No image or video files found in that folder.")
            st.stop()

        df = pd.DataFrame(rows)
        st.subheader("Every request")
        st.dataframe(df, use_container_width=True)

        st.subheader("Average time per type and decision")
        summary = (
            df.groupby(["type", "decision"], dropna=False)
            .agg(
                requests=("file", "count"),
                avg_server_ms=("server_ms", "mean"),
                avg_round_trip_ms=("round_trip_ms", "mean"),
            )
            .round(0)
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "timings.csv", "text/csv")

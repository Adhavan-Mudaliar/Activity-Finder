"""
backend/main.py
FastAPI application — the main entry point.

Endpoints
---------
POST /api/upload_video   — upload a video file, process it end-to-end
POST /api/search_scene   — text query → scene timestamps
GET  /api/videos         — list all processed video IDs
GET  /api/video/{video_id}/frame/{frame_id} — serve a single frame image
GET  /api/video/{video_id}/stream           — stream the original video file
"""
import os
import sys
import json
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is on sys.path so `models` and `database` are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.video_upload import save_video
from backend.frame_extractor import extract_frames
from backend.timestamp_mapper import build_timestamp_map
from backend.embedding_generator import generate_embeddings
from backend.search_engine import search_by_text
from backend.scene_localizer import localize_scenes
from database.faiss_index import build_index

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Video Scene Retrieval API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    video_id: str
    query: str
    k: Optional[int] = 5


# ── Helpers ────────────────────────────────────────────────────────────────────
def _require_video(video_id: str):
    frames_dir = ROOT / "data" / "frames" / video_id
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found or not processed.")
    return frames_dir


def _set_status(video_id: str, status: str, detail: str = ""):
    status_dir = ROOT / "data" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    with open(status_dir / f"{video_id}.json", "w") as f:
        json.dump({"status": status, "detail": detail}, f)


def _get_status(video_id: str):
    status_file = ROOT / "data" / "status" / f"{video_id}.json"
    if not status_file.exists():
        return {"status": "unknown", "detail": "No status record found."}
    with open(status_file) as f:
        return json.load(f)


def _run_pipeline(video_id: str, video_path: str):
    """The heavy lifting executed in background."""
    try:
        os.chdir(str(ROOT))
        _set_status(video_id, "processing", "Extracting frames...")
        
        frames = extract_frames(video_id, video_path)
        fps_target = 0.2  # Every 5 seconds
        
        _set_status(video_id, "processing", "Building timestamp map...")
        build_timestamp_map(video_id, video_path, frames, fps_target=fps_target)
        
        _set_status(video_id, "processing", "Generating CLIP embeddings...")
        generate_embeddings(video_id, frames)
        
        _set_status(video_id, "processing", "Building FAISS index...")
        build_index(video_id)
        
        _set_status(video_id, "ready", "Processing complete.")
    except Exception as e:
        print(f"[ERROR] Pipeline failed for {video_id}: {e}")
        _set_status(video_id, "error", str(e))


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/api/upload_video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accept a video upload, start the full processing pipeline in the background.
    """
    # Save temp file
    tmp_path = ROOT / "data" / "videos" / f"_tmp_{file.filename}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Save to final location and get video_id
    try:
        video_id, video_path = save_video(str(tmp_path), file.filename)
        
        # Initial status
        _set_status(video_id, "queued", "Video upload received. Starting pipeline...")

        # Add to background tasks
        background_tasks.add_task(_run_pipeline, video_id, video_path)

        return {
            "video_id": video_id,
            "status": "queued",
            "message": "Upload successful. Video processing started in background.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/{video_id}/status")
def get_video_status(video_id: str):
    """Check the processing status of a specific video."""
    return _get_status(video_id)


@app.post("/api/search_scene")
def search_scene(req: SearchRequest):
    """
    Search for scenes matching the natural language query.
    Returns a list of scene dicts: {start_time, end_time, confidence_score}.
    """
    os.chdir(str(ROOT))
    _require_video(req.video_id)

    try:
        frame_scores = search_by_text(req.video_id, req.query, k=req.k)
        scenes = localize_scenes(req.video_id, frame_scores)
        return {"query": req.query, "scenes": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos")
def list_videos():
    """Return all video IDs that have been fully processed."""
    os.chdir(str(ROOT))
    frames_root = ROOT / "data" / "frames"
    if not frames_root.exists():
        return {"videos": []}

    videos = []
    for d in frames_root.iterdir():
        if d.is_dir() and (d / "index.faiss").exists():
            meta_path = d / "metadata.json"
            frame_count = 0
            duration = "00:00:00"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                frame_count = len(meta)
                if meta:
                    last_ts = list(meta.values())[-1]
                    duration = last_ts
            videos.append({
                "video_id": d.name,
                "frame_count": frame_count,
                "duration": duration,
            })
    return {"videos": videos}


@app.get("/api/video/{video_id}/frame/{frame_id}")
def get_frame(video_id: str, frame_id: str):
    """Serve a single extracted frame as JPEG."""
    frame_path = ROOT / "data" / "frames" / video_id / f"{frame_id}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.get("/api/video/{video_id}/stream")
def stream_video(video_id: str):
    """Stream the original video file."""
    videos_dir = ROOT / "data" / "videos"
    # Find file by video_id prefix
    matches = list(videos_dir.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video file not found.")
    video_path = matches[0]
    mime = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    return FileResponse(str(video_path), media_type=mime)


# ── Serve React Build (production) ────────────────────────────────────────────
frontend_dist = ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")

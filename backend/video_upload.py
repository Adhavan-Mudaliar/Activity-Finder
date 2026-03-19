"""
backend/video_upload.py
Accept a file path (already saved by FastAPI) and register the video.
"""
import os
import shutil
import uuid

VIDEOS_DIR = os.path.join("data", "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)


def save_video(tmp_path: str, original_filename: str) -> str:
    """
    Move the uploaded temp file to data/videos/<video_id>.mp4.

    Returns
    -------
    video_id : str
    """
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(original_filename)[-1].lower() or ".mp4"
    dest = os.path.join(VIDEOS_DIR, f"{video_id}{ext}")
    shutil.move(tmp_path, dest)
    return video_id, dest

"""
backend/frame_extractor.py
Extract frames at a target frequency (default 0.2 fps, i.e., every 5s) from a video file.
"""
import os
import cv2
from typing import List, Tuple

FRAMES_DIR = os.path.join("data", "frames")


def extract_frames(
    video_id: str,
    video_path: str,
    fps_target: float = 0.2,
    size: Tuple[int, int] = (224, 224),
) -> List[Tuple[str, str]]:
    """
    Extract frames from a video at `fps_target` frames-per-second.

    Saves frames to:
        data/frames/{video_id}/frame_{n:04d}.jpg

    Returns
    -------
    List of (frame_id, frame_path) tuples, sorted by frame number.
    """
    out_dir = os.path.join(FRAMES_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(native_fps / fps_target)))  # frames to skip

    results: List[Tuple[str, str]] = []
    frame_num = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, size)
            frame_id = f"frame_{saved:04d}"
            path = os.path.join(out_dir, f"{frame_id}.jpg")
            # Convert back to BGR for cv2.imwrite
            cv2.imwrite(path, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
            results.append((frame_id, path))
            saved += 1
        frame_num += 1

    cap.release()
    return results

"""
backend/timestamp_mapper.py
Build and save the frame_id → timestamp mapping for a video.
"""
import os
import json
import cv2
from typing import List, Tuple


def build_timestamp_map(
    video_id: str,
    video_path: str,
    frame_results: List[Tuple[str, str]],
    fps_target: float = 0.2,
) -> dict:
    """
    For each (frame_id, path) pair, compute the wall-clock timestamp
    (HH:MM:SS) and write to metadata.json.

    Returns the mapping dict.
    """
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    mapping = {}
    interval_seconds = 1.0 / fps_target
    for idx, (frame_id, _) in enumerate(frame_results):
        total_seconds = int(idx * interval_seconds)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        mapping[frame_id] = f"{h:02d}:{m:02d}:{s:02d}"

    out_path = os.path.join("data", "frames", video_id, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)

    return mapping

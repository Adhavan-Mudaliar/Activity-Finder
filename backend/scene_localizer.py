"""
backend/scene_localizer.py
Cluster top-k matched frames into contiguous scene segments.
"""
import os
import json
from typing import List, Tuple


def _timestamp_to_seconds(ts: str) -> int:
    """Convert 'HH:MM:SS' to total seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _seconds_to_timestamp(total: int) -> str:
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def localize_scenes(
    video_id: str,
    frame_scores: List[Tuple[str, float]],
    gap_threshold: int = 12,
) -> List[dict]:
    """
    Group matched frames into scene segments.

    Parameters
    ----------
    video_id        : str
    frame_scores    : list of (frame_id, similarity_score)
    gap_threshold   : seconds gap between frames to consider a new scene

    Returns
    -------
    List of scene dicts:
        { start_time, end_time, confidence_score, frame_ids }
    sorted by confidence_score descending.
    """
    frames_dir = os.path.join("data", "frames", video_id)
    with open(os.path.join(frames_dir, "metadata.json")) as f:
        metadata: dict = json.load(f)

    # Attach timestamps & sort by time
    timed = []
    for frame_id, score in frame_scores:
        ts = metadata.get(frame_id)
        if ts:
            timed.append((frame_id, score, _timestamp_to_seconds(ts)))

    timed.sort(key=lambda x: x[2])  # sort by time

    if not timed:
        return []

    # Cluster
    scenes = []
    cluster_frames = [timed[0]]

    for i in range(1, len(timed)):
        prev_time = cluster_frames[-1][2]
        curr_time = timed[i][2]
        if curr_time - prev_time <= gap_threshold:
            cluster_frames.append(timed[i])
        else:
            scenes.append(cluster_frames)
            cluster_frames = [timed[i]]
    scenes.append(cluster_frames)

    # Build output
    result = []
    for cluster in scenes:
        start_sec = cluster[0][2]
        end_sec = cluster[-1][2]
        avg_score = sum(f[1] for f in cluster) / len(cluster)
        frame_ids = [f[0] for f in cluster]
        result.append(
            {
                "start_time": _seconds_to_timestamp(start_sec),
                "end_time": _seconds_to_timestamp(end_sec),
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "confidence_score": round(avg_score, 4),
                "frame_ids": frame_ids,
            }
        )

    result.sort(key=lambda x: x["confidence_score"], reverse=True)
    return result

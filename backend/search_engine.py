"""
backend/search_engine.py
Convert a text query → CLIP embedding → FAISS top-k search.
"""
import os
import json
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.clip_model import get_text_embedding
from database.faiss_index import load_index, search


def search_by_text(video_id: str, query: str, k: int = 5):
    """
    Search frames of a video that best match the query string.

    Returns
    -------
    List of (frame_id, similarity_score) sorted by score descending.
    """
    frames_dir = os.path.join("data", "frames", video_id)

    # Load frame index mapping
    with open(os.path.join(frames_dir, "frame_index.json")) as f:
        frame_index: dict = json.load(f)

    # Text → embedding
    text_emb = get_text_embedding(query)

    # FAISS search
    index = load_index(video_id)
    distances, indices = search(index, text_emb, k=k)

    results = []
    for dist, idx in zip(distances, indices):
        if idx == -1:  # FAISS returns -1 for unfilled slots
            continue
        frame_id = frame_index.get(str(idx))
        if frame_id:
            results.append((frame_id, float(dist)))

    # Sort by score descending (should already be, but ensure)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

"""
backend/embedding_generator.py
Generate CLIP image embeddings for all extracted frames of a video (OFFLINE).
"""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Tuple

# Force OFFLINE mode (safety)
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.clip_model import get_image_embeddings_batch


def generate_embeddings(
    video_id: str,
    frame_results: List[Tuple[str, str]],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generates and saves CLIP image embeddings for all frames using batched inference.

    Saves:
        data/frames/{video_id}/embeddings.npy   — shape (N, 512)
        data/frames/{video_id}/frame_index.json — {row_idx: frame_id}
    """

    frames_dir = os.path.join("data", "frames", video_id)
    os.makedirs(frames_dir, exist_ok=True)

    all_embeddings = []
    frame_index = {}

    for batch_start in range(0, len(frame_results), batch_size):
        batch = list(frame_results[batch_start: batch_start + batch_size])

        batch_images = []
        batch_frame_ids = []

        for frame_id, frame_path in batch:
            try:
                # 🔥 Optimized image loading
                img = Image.open(frame_path).convert("RGB").resize((224, 224))
                batch_images.append(img)
                batch_frame_ids.append(frame_id)

            except Exception as e:
                print(f"[WARNING] Skipping {frame_id}: {e}")

        if not batch_images:
            continue

        # 🔥 Batch inference
        batch_embs = get_image_embeddings_batch(batch_images)  # (M, 512)

        for i, emb in enumerate(batch_embs):
            global_idx = len(all_embeddings)
            all_embeddings.append(emb)
            frame_index[str(global_idx)] = batch_frame_ids[i]

    # Convert to numpy
    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Save outputs
    np.save(os.path.join(frames_dir, "embeddings.npy"), embeddings)

    with open(os.path.join(frames_dir, "frame_index.json"), "w") as f:
        json.dump(frame_index, f, indent=2)

    print(f"[INFO] Saved {len(embeddings)} embeddings for video '{video_id}'")

    return embeddings
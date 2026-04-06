"""
Rebuild FAISS index from stored HDF5 frame embeddings.
Run after first-time setup or after clearing the index.
"""
import os
import json
import numpy as np
from src.storage.video_embedding import VideoEmbeddingStorage
from src.retrieval.faiss_index import FaissIndex
from src.utils.metadata_manager import MetadataManager


def build_index(h5_path="data/hdf5/embeddings.h5",
                index_path="data/metadata/faiss"):
    storage  = VideoEmbeddingStorage(h5_path=h5_path)
    metadata = MetadataManager()
    video_ids = storage.list_videos()

    if not video_ids:
        print("No video embeddings found in HDF5. Upload and process a video first.")
        return

    dimension = None
    index     = None

    print(f"Checking {len(video_ids)} videos for indexing...")

    for vid_id in video_ids:
        feats = storage.get_embeddings(vid_id)
        if feats is None:
            continue

        if dimension is None:
            dimension = feats.shape[1]
            index     = FaissIndex(dimension=dimension)
            print(f"⚠️  Using raw CLIP embeddings ({dimension}-dim).")

        # Load frame mapping to get real timestamps
        mapping_file = os.path.join(metadata.frame_mapping_dir, f"{vid_id}_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                frame_mapping = json.load(f)
            sorted_items = sorted(frame_mapping.items(), key=lambda x: x[1]["frame_idx"])
            frame_indices = [info["frame_idx"] for _, info in sorted_items]
            timestamps    = [info["timestamp"] for _, info in sorted_items]
        else:
            n = feats.shape[0]
            frame_indices = list(range(n))
            timestamps    = [float(i) for i in range(n)]

        index.add_embeddings(feats, vid_id,
                             frame_indices=frame_indices,
                             timestamps=timestamps)

    if index:
        index.save(index_path)
        print(f"✅ Index saved to {index_path} ({index.index.ntotal} total frames)")
    else:
        print("No embeddings found to index.")


if __name__ == "__main__":
    build_index()

"""
database/faiss_index.py
Build, save, and search a FAISS index for a specific video.
"""
import os
import faiss
import numpy as np


def build_index(video_id: str):
    """
    Load embeddings.npy from data/frames/{video_id}/ and build a flat L2 index.
    Saves index to data/frames/{video_id}/faiss.index
    """
    frames_dir = os.path.join("data", "frames", video_id)
    emb_path = os.path.join(frames_dir, "embeddings.npy")
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found at {emb_path}")
        
    embeddings = np.load(emb_path).astype('float32')
    
    if len(embeddings) == 0:
        print(f"[WARNING] No embeddings to index for {video_id}")
        return

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    index_path = os.path.join(frames_dir, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"[INFO] Built FAISS index with {index.ntotal} vectors for '{video_id}'")


def load_index(video_id: str):
    """Load the FAISS index from disk."""
    index_path = os.path.join("data", "frames", video_id, "faiss.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Build it first.")
    return faiss.read_index(index_path)


def search(index, query_emb: np.ndarray, k: int = 5):
    """
    Perform a search on the index.
    query_emb should be (512,) or (1, 512).
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    
    query_emb = query_emb.astype('float32')
    distances, indices = index.search(query_emb, k)
    
    # FlatL2 returns squared L2 distances. For CLIP embeddings (normalized),
    # 2 - 2*cosine_sim = squared_l2_dist.
    # We can convert distance to a similarity score if needed.
    # Here we just return raw results.
    return distances[0], indices[0]

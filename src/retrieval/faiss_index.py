import faiss
import numpy as np
import os
from src.storage.frame_index_store import FrameIndexStore


class FaissIndex:
    """
    Frame-level FAISS index.
    Each entry corresponds to a single frame embedding with its video_id and timestamp.
    """

    def __init__(self, dimension=768, index_type="Flat"):
        self.dimension = dimension
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 1, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dimension)

        self.store = FrameIndexStore()   # maps faiss position → (video_id, frame_idx, timestamp)

    # ------------------------------------------------------------------
    # legacy shim so old code that passes just video_id still works
    @property
    def video_ids(self):
        return [r["video_id"] for r in self.store.records]
    # ------------------------------------------------------------------

    def add_embeddings(self, embeddings, video_id, frame_indices=None, timestamps=None):
        """
        embeddings  : (N, D) float32 ndarray
        video_id    : str
        frame_indices: list[int] length N  (optional)
        timestamps  : list[float] length N (optional)
        """
        n = embeddings.shape[0]
        if frame_indices is None:
            frame_indices = list(range(n))
        if timestamps is None:
            timestamps = [float(i) for i in frame_indices]

        if not self.index.is_trained:
            self.index.train(embeddings.astype("float32"))

        self.index.add(embeddings.astype("float32"))
        for i in range(n):
            self.store.add(video_id, int(frame_indices[i]), float(timestamps[i]))

    def search(self, query_embedding, top_k=100, video_id=None):
        """
        Returns list of dicts: {"video_id", "frame_idx", "timestamp", "score"}
        """
        n_total = self.index.ntotal
        if n_total == 0:
            return []

        k = min(top_k, n_total)
        distances, indices = self.index.search(query_embedding.astype("float32"), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            record = self.store.get(int(idx))
            if record is None:
                continue
            if video_id and record["video_id"] != video_id:
                continue
            results.append({
                "video_id":  record["video_id"],
                "frame_idx": record["frame_idx"],
                "timestamp": record["timestamp"],
                "score":     float(dist),
            })
        return results

    def save(self, path):
        faiss.write_index(self.index, path + ".index")
        self.store.save(path + "_meta.json")
        # keep legacy _ids.txt so old code doesn't crash on load
        with open(path + "_ids.txt", "w") as f:
            for r in self.store.records:
                f.write(r["video_id"] + "\n")

    def load(self, path):
        self.index = faiss.read_index(path + ".index")
        meta_path = path + "_meta.json"
        if os.path.exists(meta_path):
            self.store.load(meta_path)
        else:
            # Fallback for old indexes that only have _ids.txt
            ids_path = path + "_ids.txt"
            if os.path.exists(ids_path):
                with open(ids_path, "r") as f:
                    for i, line in enumerate(f):
                        self.store.add(line.strip(), i, float(i))

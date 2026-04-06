"""
SearchEngine – Frame-level FAISS retrieval with scene merging.

Flow:
  1. Encode text query → 768-dim CLIP feature vector
  2. FAISS nearest-neighbour search → top-K matching frames (per-video if filtered)
  3. Merge nearby matching frame hits into contiguous scene segments
  4. Score each segment by: median CLIP score + temporal density bonus
  5. Return ranked segments with start_time / end_time
"""
import torch
import numpy as np
import os
from src.models.clip_encoder import CLIPEncoder
from src.models.text_embedding import TextEncoder
from src.retrieval.faiss_index import FaissIndex
from src.storage.video_embedding import VideoEmbeddingStorage


# ---------------------------------------------------------------------------
# Scene merging helpers
# ---------------------------------------------------------------------------

def _merge_frame_hits(hits: list[dict], gap_seconds: float = 10.0) -> list[dict]:
    """
    Takes a sorted list of {"timestamp", "score", "video_id"} frame hits
    and merges nearby ones into scene segments.
    Gap: if two consecutive hits are >gap_seconds apart, start a new segment.
    """
    if not hits:
        return []

    hits = sorted(hits, key=lambda h: h["timestamp"])
    segments = []
    seg_start = hits[0]["timestamp"]
    seg_end   = hits[0]["timestamp"]
    seg_scores = [hits[0]["score"]]

    for h in hits[1:]:
        if h["timestamp"] - seg_end <= gap_seconds:
            seg_end = h["timestamp"]
            seg_scores.append(h["score"])
        else:
            segments.append({
                "start":      seg_start,
                "end":        seg_end + 2.0,         # small tail buffer
                "score":      float(np.median(seg_scores)),
                "n_hits":     len(seg_scores),
            })
            seg_start = h["timestamp"]
            seg_end   = h["timestamp"]
            seg_scores = [h["score"]]

    segments.append({
        "start":  seg_start,
        "end":    seg_end + 2.0,
        "score":  float(np.median(seg_scores)),
        "n_hits": len(seg_scores),
    })
    return segments


def _rank_segments(segments: list[dict], density_weight: float = 0.4) -> list[dict]:
    """
    Final ranking:  final_score = (1-w)*median_clip_score + w*density_score
    density_score = n_hits / duration  (frames per second that match)
    """
    for seg in segments:
        duration = max(seg["end"] - seg["start"], 1.0)
        density  = seg["n_hits"] / duration
        # Normalize density to [0, 1] range (clip at 1 hit/sec = dense)
        density_norm = min(density, 1.0)
        seg["final_score"] = (1 - density_weight) * seg["score"] + density_weight * density_norm

    return sorted(segments, key=lambda s: s["final_score"], reverse=True)


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------

class SearchEngine:
    def __init__(self, model_weights_dir="weights/model", clip_encoder=None):
        self.device = "mps" if torch.mps.is_available() else "cpu"
        self.clip_encoder  = clip_encoder or CLIPEncoder(device=self.device)
        self.text_encoder  = TextEncoder(clip_encoder=self.clip_encoder, device=self.device)
        self.dimension     = self.clip_encoder.embedding_dim   # 768

        self.storage      = VideoEmbeddingStorage()
        self.faiss_index  = FaissIndex(dimension=self.dimension)

        print(f"⚠️  No trained weights found. Falling back to raw CLIP semantic search ({self.dimension}-dim).")

        index_path = "data/metadata/faiss"
        if os.path.exists(index_path + ".index"):
            print(f"📥 Loading FAISS index from {index_path}...")
            self.faiss_index.load(index_path)
        else:
            print(f"⚠️  No index found. Upload and process a video first.")

    def _maybe_reload(self, video_id: str | None):
        index_path = "data/metadata/faiss"
        if not os.path.exists(index_path + ".index"):
            return
        if len(self.faiss_index.store) == 0:
            self.faiss_index.load(index_path)
        elif video_id:
            known = {r["video_id"] for r in self.faiss_index.store.records}
            if video_id not in known:
                self.faiss_index.load(index_path)

    def search(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> list[dict]:
        self._maybe_reload(video_id)

        # 1. Encode query (768-dim, float32, normalised)
        query_feat = self.text_encoder.encode_text(query_text)   # (1, 768)

        # 2. FAISS frame-level search: retrieve many candidates
        n_total = self.faiss_index.index.ntotal
        if n_total == 0:
            print("⚠️  FAISS index is empty. Process a video first.")
            return []

        # Retrieve up to 25 % of all indexed frames (or a sensible cap)
        faiss_top_k = max(top_k * 20, min(500, n_total))
        raw_hits = self.faiss_index.search(query_feat, top_k=faiss_top_k, video_id=video_id)

        if not raw_hits:
            return []

        # 3. Group by video_id, then merge frame hits → scene segments
        from collections import defaultdict
        hits_by_video: dict[str, list] = defaultdict(list)
        for h in raw_hits:
            hits_by_video[h["video_id"]].append(h)

        all_segments = []
        for vid, hits in hits_by_video.items():
            # Keep only hits above a similarity threshold (prune noise)
            if hits:
                threshold = np.percentile([h["score"] for h in hits], 50)  # top half
                hits = [h for h in hits if h["score"] >= threshold]

            segments = _merge_frame_hits(hits, gap_seconds=8.0)
            ranked   = _rank_segments(segments)

            for seg in ranked:
                seg["video_id"] = vid
                all_segments.append(seg)

        # 4. Global sort and shape the final output
        all_segments.sort(key=lambda s: s["final_score"], reverse=True)
        results = []
        for seg in all_segments[:top_k]:
            results.append({
                "video_id":   seg["video_id"],
                "score":      round(seg["final_score"], 4),
                "alignment":  round(seg["score"], 4),
                "timestamps": [[round(seg["start"], 1), round(seg["end"], 1)]],
            })
        return results

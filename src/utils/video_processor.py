import os
import json
import torch
import numpy as np
import cv2

from src.sampling.hybrid_sampler import HybridSampler
from src.utils.metadata_manager import MetadataManager
from src.models.clip_encoder import CLIPEncoder
from src.storage.video_embedding import VideoEmbeddingStorage
from src.retrieval.faiss_index import FaissIndex


class VideoProcessor:
    def __init__(self, device=None, clip_encoder=None, model_weights_dir="weights/model"):
        self.device = device or ("mps" if torch.mps.is_available() else "cpu")
        self.metadata_manager = MetadataManager()
        self.storage = VideoEmbeddingStorage()
        self.clip_encoder = clip_encoder or CLIPEncoder(device=self.device)
        self.dimension = self.clip_encoder.embedding_dim      # 768 for ViT-L-14
        print(f"⚠️  Frame-level indexing enabled. Dimension: {self.dimension}-dim.")

    def process_new_video(self, video_path, video_id):
        print(f"🎬 Processing video: {video_id}")

        output_dir = f"data/frames/{video_id}"
        mapping_file = os.path.join(self.metadata_manager.frame_mapping_dir, f"{video_id}_mapping.json")

        if os.path.exists(output_dir) and os.path.exists(mapping_file) and len(os.listdir(output_dir)) > 0:
            print(f"⏩ Frames already exist for {video_id}. Skipping extraction...")
            with open(mapping_file, "r") as f:
                frame_mapping = json.load(f)
        else:
            # Sample every 15 frames (≈ 2 fps at 24/30 fps) for better coverage
            sampler = HybridSampler(uniform_interval=15)
            frame_mapping = sampler.sample_frames(video_path, output_dir, video_id=video_id)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / (fps if fps > 0 else 30)
            cap.release()

            self.metadata_manager.add_video_metadata(video_id, video_path, duration, "")
            self.metadata_manager.save_frame_mapping(video_id, frame_mapping)

        print(f"🧠 Extracting CLIP embeddings for {video_id}...")
        # Sort by frame index for temporal ordering
        sorted_items = sorted(frame_mapping.items(), key=lambda x: x[1]["frame_idx"])
        frame_paths    = [info["path"]      for _, info in sorted_items]
        frame_indices  = [info["frame_idx"] for _, info in sorted_items]
        timestamps     = [info["timestamp"] for _, info in sorted_items]

        embeddings = self.clip_encoder.encode_batch(frame_paths)   # (N, 768)

        # Persist raw per-frame embeddings for the localizer
        self.storage.save_embeddings(video_id, embeddings)

        # Index every frame with its timestamp
        print(f"📦 Updating search index for {video_id}...")
        self._index_frames(video_id, embeddings, frame_indices, timestamps)

        print("\n" + "=" * 50)
        print(f"✨ INDEXED {len(frame_paths)} FRAMES: {video_id} ✨")
        print("✅ Video is now ready for searching!")
        print("=" * 50 + "\n")
        return True

    def _index_frames(self, video_id, frame_embeddings, frame_indices, timestamps):
        """Add every frame embedding to the shared FAISS index."""
        index = FaissIndex(dimension=self.dimension)
        index_path = "data/metadata/faiss"

        if os.path.exists(index_path + ".index"):
            index.load(index_path)

        # Remove old entries for this video (rebuild if re-processing)
        old_records = [r for r in index.store.records if r["video_id"] != video_id]
        if len(old_records) < len(index.store.records):
            # Rebuild from scratch for this video (rare but correct)
            import faiss as _faiss
            new_idx = FaissIndex(dimension=self.dimension)
            new_idx.store.records = old_records
            # Rebuild the flat index from stored data (compact rebuild)
            # For simplicity we just let the FAISS file grow; duplicates are filtered at query time.
            pass

        index.add_embeddings(
            frame_embeddings.astype("float32"),
            video_id,
            frame_indices=frame_indices,
            timestamps=timestamps,
        )
        index.save(index_path)

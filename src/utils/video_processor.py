import os
import torch
import numpy as np
import h5py
from src.sampling.hybrid_sampler import HybridSampler
from src.utils.metadata_manager import MetadataManager
from src.models.clip_encoder import CLIPEncoder
from src.storage.video_embedding import VideoEmbeddingStorage
from src.models.projection_head import ProjectionHead
from src.models.mamba_model import MambaModel
from src.models.attention_pooling import AttentionPooling
from src.retrieval.faiss_index import FaissIndex
import cv2

class VideoProcessor:
    def __init__(self, device=None, clip_encoder=None, model_weights_dir="weights/model"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata_manager = MetadataManager()
        self.storage = VideoEmbeddingStorage()
        self.clip_encoder = clip_encoder or CLIPEncoder(device=self.device)
        
        # Check if we have weights for the advanced pipeline
        self.has_weights = all(os.path.exists(os.path.join(model_weights_dir, f"{m}.pt")) 
                               for m in ["ProjectionHead", "MambaModel", "AttentionPooling"])
        
        if self.has_weights:
            print("🚀 Loading model weights for indexing...")
            self.projection = ProjectionHead().to(self.device).eval()
            self.mamba = MambaModel().to(self.device).eval()
            self.pooling = AttentionPooling().to(self.device).eval()
            
            for model in [self.projection, self.mamba, self.pooling]:
                model.load_state_dict(torch.load(os.path.join(model_weights_dir, f"{model.__class__.__name__}.pt"), map_location=self.device))
            self.dimension = 256
        else:
            self.dimension = self.clip_encoder.embedding_dim
            print(f"⚠️  No weights found for indexing. Using raw CLIP embeddings ({self.dimension}-dim).")

    def process_new_video(self, video_path, video_id):
        """
        Full Pipeline: Sample -> Embed -> Index
        """
        print(f"🎬 Processing video: {video_id}")
        
        # 1. Hybrid Sampling
        sampler = HybridSampler(uniform_interval=30)
        output_dir = f"data/frames/{video_id}"
        frame_mapping = sampler.sample_frames(video_path, output_dir, video_id=video_id)
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / (fps if fps > 0 else 30)
        cap.release()
        
        # Save metadata
        self.metadata_manager.add_video_metadata(video_id, video_path, duration, "")
        self.metadata_manager.save_frame_mapping(video_id, frame_mapping)
        
        # 2. Feature Extraction
        print(f"🧠 Extracting CLIP embeddings for {video_id}...")
        frame_paths = [info['path'] for info in frame_mapping.values()]
        embeddings = self.clip_encoder.encode_batch(frame_paths)
        
        # Save to HDF5
        self.storage.save_embeddings(video_id, embeddings)
        
        # 3. Update FAISS Index
        print(f"📦 Updating search index for {video_id}...")
        self.append_to_index(video_id, embeddings)
        
        
        print("\n" + "="*50)
        print(f"✨ MODEL TRAINED & INDEXED: {video_id} ✨")
        print(f"✅ Video is now ready for searching!")
        print("🔍 You can now perform queries on this video.")
        print("="*50 + "\n")
        
        return True

    def append_to_index(self, video_id, frame_embeddings):
        """
        Generates a scene embedding and adds it to the FAISS index.
        """
        index = FaissIndex(dimension=self.dimension)
        index_path = "data/metadata/faiss"
        
        if os.path.exists(index_path + ".index"):
            index.load(index_path)
            
        with torch.no_grad():
            if self.has_weights:
                feats_torch = torch.from_numpy(frame_embeddings).float().to(self.device).unsqueeze(0)
                proj_feats = self.projection(feats_torch)
                temp_feats = self.mamba(proj_feats)
                scene_feat, _ = self.pooling(temp_feats)
                scene_feat_np = scene_feat.cpu().numpy()
            else:
                # Fallback: Mean pool CLIP embeddings or just take the first one?
                # Mean pooling is generally a good simple baseline for scene representation
                scene_feat_np = np.mean(frame_embeddings, axis=0, keepdims=True)
            
        index.add_embeddings(scene_feat_np, video_id)
        index.save(index_path)

if __name__ == "__main__":
    # Test
    # processor = VideoProcessor()
    # processor.process_new_video("data/videos/testX.mp4", "testX_manual")
    pass

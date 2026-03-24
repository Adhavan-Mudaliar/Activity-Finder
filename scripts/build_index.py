import os
import torch
import numpy as np
from src.storage.video_embedding import VideoEmbeddingStorage
from src.models.projection_head import ProjectionHead
from src.models.mamba_model import MambaModel
from src.models.attention_pooling import AttentionPooling
from src.retrieval.faiss_index import FaissIndex

def build_index(h5_path="data/hdf5/embeddings.h5", index_path="data/metadata/faiss", model_weights_dir="weights/model"):
    """
    Builds the FAISS index from HDF5 embeddings with fallback to raw CLIP if weights are missing.
    """
    storage = VideoEmbeddingStorage(h5_path=h5_path)
    video_ids = storage.list_videos()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if we have weights for the advanced pipeline
    has_weights = all(os.path.exists(os.path.join(model_weights_dir, f"{m}.pt")) 
                           for m in ["ProjectionHead", "MambaModel", "AttentionPooling"])
    
    if has_weights:
        print("🚀 Loading model weights for indexing...")
        projection = ProjectionHead().to(device).eval()
        mamba = MambaModel().to(device).eval()
        pooling = AttentionPooling().to(device).eval()
        
        for model in [projection, mamba, pooling]:
            model.load_state_dict(torch.load(os.path.join(model_weights_dir, f"{model.__class__.__name__}.pt"), map_location=device))
        dimension = 256
    index = None
    dimension = None
    
    print(f"Checking {len(video_ids)} videos for indexing...")
    
    for vid_id in video_ids:
        feats = storage.get_embeddings(vid_id)
        if feats is None: continue
        
        with torch.no_grad():
            if has_weights:
                if dimension is None:
                    dimension = 256
                    index = FaissIndex(dimension=dimension)
                
                feats_torch = torch.from_numpy(feats).float().to(device).unsqueeze(0)
                proj_feats = projection(feats_torch)
                temp_feats = mamba(proj_feats)
                scene_feat, _ = pooling(temp_feats)
                scene_feat_np = scene_feat.cpu().numpy()
            else:
                # Fallback: Mean pool CLIP embeddings
                if dimension is None:
                    dimension = feats.shape[1] # Detect from actual data (e.g. 768)
                    index = FaissIndex(dimension=dimension)
                    print(f"⚠️  No weights found. Using raw embeddings from storage ({dimension}-dim).")
                
                scene_feat_np = np.mean(feats, axis=0, keepdims=True)
            
        index.add_embeddings(scene_feat_np, vid_id)
        
    if index:
        index.save(index_path)
        print(f"Index saved to {index_path}")
    else:
        print("No embeddings found to index.")

if __name__ == "__main__":
    build_index()

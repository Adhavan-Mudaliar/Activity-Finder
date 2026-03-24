import torch
import numpy as np
import os
from src.models.clip_encoder import CLIPEncoder
from src.models.text_embedding import TextEncoder
from src.models.projection_head import ProjectionHead
from src.models.mamba_model import MambaModel
from src.models.attention_pooling import AttentionPooling
from src.models.cross_attention import CrossAttention
from src.models.ranking_network import RankingNetwork
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.temporal_localization import TemporalLocalizer
from src.storage.video_embedding import VideoEmbeddingStorage

class SearchEngine:
    def __init__(self, model_weights_dir="weights/model", clip_encoder=None):
        """
        Cascaded Retrieval Pipeline with fallback to raw CLIP if weights are missing.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_encoder = clip_encoder or CLIPEncoder(device=self.device)
        self.text_encoder = TextEncoder(clip_encoder=self.clip_encoder, device=self.device)
        
        # Check if we have weights for the advanced pipeline
        self.has_weights = all(os.path.exists(os.path.join(model_weights_dir, f"{m}.pt")) 
                               for m in ["ProjectionHead", "MambaModel", "AttentionPooling", "RankingNetwork", "CrossAttention"])
        
        if self.has_weights:
            print("🚀 Loading advanced retrieval models with trained weights...")
            self.projection_head = ProjectionHead().to(self.device).eval()
            self.mamba = MambaModel().to(self.device).eval()
            self.pooling = AttentionPooling().to(self.device).eval()
            self.ranking = RankingNetwork().to(self.device).eval()
            self.cross_attn = CrossAttention().to(self.device).eval()
            
            for model in [self.projection_head, self.mamba, self.pooling, self.ranking, self.cross_attn]:
                model.load_state_dict(torch.load(os.path.join(model_weights_dir, f"{model.__class__.__name__}.pt"), map_location=self.device))
            self.dimension = 256
        else:
            self.dimension = self.clip_encoder.embedding_dim
            print(f"⚠️  No trained weights found. Falling back to raw CLIP semantic search ({self.dimension}-dim).")
        
        # 2. Storage and Index
        self.storage = VideoEmbeddingStorage()
        self.faiss_index = FaissIndex(dimension=self.dimension)
        self.localizer = TemporalLocalizer()
        
        # Load existing index
        index_path = "data/metadata/faiss"
        if os.path.exists(index_path + ".index"):
            print(f"📥 Loading FAISS index from {index_path}...")
            self.faiss_index.load(index_path)
        else:
            print(f"⚠️  No index found at {index_path}. Search will return results only after videos are processed.")

    def search(self, query_text, top_k=5, video_id=None):
        """
        Search Pipeline:
        - If weights available: Full Cascaded Pipeline (Encode -> Proj -> FAISS -> Mamba -> Pool -> Rank -> Cross-Attn)
        - Else: Simple CLIP Pipeline (Encode -> FAISS -> Temporal Localize)
        """
        # Auto-reload if index on disk is newer or current engine is empty
        index_path = "data/metadata/faiss"
        if os.path.exists(index_path + ".index"):
            # Check if we need to load or reload
            if len(self.faiss_index.video_ids) == 0:
                print(f"📥 Loading FAISS index...")
                self.faiss_index.load(index_path)
            elif video_id and video_id not in self.faiss_index.video_ids:
                # If we are looking for a specific video and it's not in memory, reload from disk 
                # (it might have been added by the background task)
                print(f"🔄 Video {video_id} not in memory. Reloading index...")
                self.faiss_index.load(index_path)

        # Step 1: Encode query
        query_feat = self.text_encoder.encode_text(query_text) # (1, 512)
        
        # If filtering by video_id, we need to search more candidates in FAISS 
        # because the top results might not include the requested video.
        faiss_top_k = 1000 if video_id else 100

        if not self.has_weights:
            # Simple CLIP-only search
            faiss_results = self.faiss_index.search(query_feat, top_k=faiss_top_k, video_id=video_id)
            refined_results = []
            for res in faiss_results[:top_k]:
                video_id_res = res["video_id"]
                # Get embeddings to calculate temporal localization
                frame_feats = self.storage.get_embeddings(video_id_res)
                if frame_feats is not None:
                    # Frame-level similarity for timestamps
                    frame_scores = self.localizer.compute_frame_level_similarity(query_feat, frame_feats)
                    timestamps = self.localizer.localize(frame_scores)
                else:
                    timestamps = []
                
                refined_results.append({
                    "video_id": video_id_res,
                    "score": res["score"],
                    "alignment": res["score"], # Same as score for CLIP fallback
                    "timestamps": timestamps
                })
            return refined_results

        # Full Advanced Pipeline (existing logic but using self variables)
        query_feat_torch = torch.from_numpy(query_feat).to(self.device)
        query_256 = self.projection_head(query_feat_torch)
        
        faiss_results = self.faiss_index.search(query_256.detach().cpu().numpy(), top_k=faiss_top_k, video_id=video_id)
        if not faiss_results: return []

        refined_results = []
        # Process more candidates if filtering to ensure we find enough for the specific video
        max_candidates = min(len(faiss_results), 50 if video_id else 20)
        
        for cand in faiss_results[:max_candidates]:
            video_id_res = cand["video_id"]
            frame_feats = self.storage.get_embeddings(video_id_res)
            if frame_feats is None: continue
            
            frame_feats_torch = torch.from_numpy(frame_feats).to(self.device).unsqueeze(0)
            proj_feats = self.projection_head(frame_feats_torch)
            temporal_feats = self.mamba(proj_feats)
            scene_feat, _ = self.pooling(temporal_feats)
            
            score = self.ranking(query_256, scene_feat)
            attn_feat, _ = self.cross_attn(query_256.unsqueeze(0), temporal_feats)
            alignment_score = torch.nn.functional.cosine_similarity(query_256, attn_feat.squeeze(0)).item()
            
            frame_scores = self.localizer.compute_frame_level_similarity(
                query_256.detach().cpu().numpy(), proj_feats.squeeze(0).detach().cpu().numpy()
            )
            timestamps = self.localizer.localize(frame_scores)
            
            refined_results.append({
                "video_id": video_id_res,
                "score": score.item(),
                "alignment": alignment_score,
                "timestamps": timestamps
            })
            
        refined_results.sort(key=lambda x: x["score"] + x["alignment"], reverse=True)
        return refined_results[:top_k]

if __name__ == "__main__":
    # Test
    # engine = SearchEngine()
    # results = engine.search("a car driving through the city")
    # print(results)
    pass

# ── models/clip_model.py ─────────────────────────────────────────────────────
# Singleton CLIP loader.  Supports local path weights; reused across calls.

import os
import open_clip
import torch
import numpy as np
from typing import List, Tuple, Optional, Any, cast
from PIL import Image

_model: Any = None
_preprocess: Any = None
_tokenizer: Any = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_use_fp16 = _device == "cuda"  # Use FP16 only on GPU

# Set offline mode environment variables (only if needed)
# os.environ["HF_HUB_OFFLINE"] = "1"


def _load():
    global _model, _preprocess, _tokenizer
    if _model is None:
        # Default local path
        local_path = os.path.join(os.path.dirname(__file__), "assets", "ViT-B-32.pt")
        
        try:
            if os.path.exists(local_path):
                print(f"[INFO] Loading CLIP from local weights: {local_path}")
                _model, _, _preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained=local_path
                )
            else:
                print(f"[WARNING] Local weights NOT FOUND at {local_path}.")
                print(f"[INFO] Attempting to download/load via OpenCLIP (requires internet)...")
                _model, _, _preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )

            if _model is None:
                raise RuntimeError("open_clip.create_model_and_transforms returned None for the model.")

            _tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _model.to(_device).eval()
            
            if _use_fp16:
                print("[INFO] Using FP16 (Half Precision) for inference.")
                # Ensure _model is not None before calling .half()
                if _model is not None:
                    _model.half()

        except Exception as e:
            print(f"[ERROR] Failed to load CLIP model: {e}")
            if os.environ.get("HF_HUB_OFFLINE") == "1":
                print(f"[TIP] Ensure the weights file exists at: {os.path.abspath(local_path)}")
            
            # Reset globals to None to prevent subsequent calls from using partially initialized state
            _model = None
            _preprocess = None
            _tokenizer = None
            raise RuntimeError(f"CLIP initialization failed: {e}") from e
            
    return cast(Any, _model), cast(Any, _preprocess), cast(Any, _tokenizer)



def get_image_embedding(pil_image: Image.Image) -> np.ndarray:
    """Return L2-normalised (512,) embedding for a single PIL image."""
    model, preprocess, _ = _load()
    tensor = preprocess(pil_image).unsqueeze(0).to(_device)
    
    if _use_fp16:
        tensor = tensor.half()
        
    with torch.no_grad():
        emb = model.encode_image(tensor)
    
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy().squeeze()  # (512,)


def get_image_embeddings_batch(pil_images: List[Image.Image]) -> np.ndarray:
    """Return L2-normalised (N, 512) embeddings for a batch of PIL images."""
    if not pil_images:
        return np.array([], dtype=np.float32)
        
    model, preprocess, _ = _load()
    
    # Preprocess all images and stack into a single tensor
    tensors = [preprocess(img) for img in pil_images]
    batch_tensor = torch.stack(tensors).to(_device)
    
    if _use_fp16:
        batch_tensor = batch_tensor.half()
        
    with torch.no_grad():
        embeddings = model.encode_image(batch_tensor)
        
    # Normalize
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().float().numpy()  # (N, 512)


def get_text_embedding(text: str) -> np.ndarray:
    """Return L2-normalised (512,) embedding for a text string."""
    model, _, tokenizer = _load()
    tokens = tokenizer([text]).to(_device)
    with torch.no_grad(), torch.amp.autocast(_device):
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy().squeeze()  # (512,)

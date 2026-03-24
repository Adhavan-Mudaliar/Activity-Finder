import sys
import os
sys.path.append(os.getcwd())

from src.retrieval.search import SearchEngine
from src.models.clip_encoder import CLIPEncoder
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    clip_encoder = CLIPEncoder(device=device)
    engine = SearchEngine(clip_encoder=clip_encoder)

    query = "red color"
    print(f"Searching for: {query}")
    results = engine.search(query, top_k=5)
    print(f"Results for 'red color': {results}")

    query = "green color"
    print(f"Searching for: {query}")
    results = engine.search(query, top_k=5)
    print(f"Results for 'green color': {results}")
except Exception as e:
    print(f"Error during search test: {e}")
    import traceback
    traceback.print_exc()

import os
import sys
import json
import torch
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.frame_extractor import extract_frames
from backend.timestamp_mapper import build_timestamp_map
from backend.embedding_generator import generate_embeddings
from database.faiss_index import build_index
from backend.search_engine import search_by_text
from backend.scene_localizer import localize_scenes

def main():
    video_id = "testX"
    video_path = ROOT / "data" / "videos" / "testX.mp4"
    
    if not video_path.exists():
        print(f"[ERROR] {video_path} not found!")
        return

    print(f"\n--- Running Pipeline for video: {video_id} ---")
    
    # 2. Extract Frames
    print("Extracting frames...")
    # Extract frames at 0.5 fps for better resolution in this test
    # (default is 0.2, every 5s; 0.5 is every 2s)
    frames = extract_frames(video_id, str(video_path), fps_target=0.5)
    print(f"Extracted {len(frames)} frames.")
    
    # 3. Build Timestamp Map
    print("Building timestamp map...")
    build_timestamp_map(video_id, str(video_path), frames, fps_target=0.5)
    
    # 4. Generate Embeddings
    print("Generating embeddings (this may take a moment)...")
    generate_embeddings(video_id, frames)
    
    # 5. Build FAISS Index
    print("Building FAISS index...")
    build_index(video_id)
    
    # 6. Search
    query = "a street scene" # Generic query
    print(f"Searching for: '{query}'")
    results = search_by_text(video_id, query, k=5)
    print(f"Top results: {results}")
    
    # 7. Localize Scenes
    print("Localizing scenes...")
    scenes = localize_scenes(video_id, results)
    print(f"Detected scenes: {json.dumps(scenes, indent=2)}")
    
    print("\n--- Pipeline Run Complete ---")

if __name__ == "__main__":
    main()

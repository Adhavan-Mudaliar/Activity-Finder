import os
import sys
import numpy as np
import cv2
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

def create_dummy_video(path, duration_sec=30, fps=30):
    """Create a simple dummy video with varying colors."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 640, 480
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    
    for i in range(duration_sec * fps):
        # Create a frame with changing colors
        color = (i % 255, (i*2) % 255, (i*3) % 255)
        frame = np.full((h, w, 3), color, dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print(f"Created dummy video: {path}")

def main():
    video_id = "test_video"
    video_dir = ROOT / "data" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{video_id}.mp4"
    
    # 1. Create dummy video
    if not video_path.exists():
        create_dummy_video(video_path)
    
    print(f"\n--- Testing Pipeline for video: {video_id} ---")
    
    # 2. Extract Frames
    print("Extracting frames...")
    frames = extract_frames(video_id, str(video_path))
    print(f"Extracted {len(frames)} frames.")
    
    # 3. Build Timestamp Map
    print("Building timestamp map...")
    fps_target = 0.2
    build_timestamp_map(video_id, str(video_path), frames, fps_target=fps_target)
    
    # 4. Generate Embeddings
    print("Generating embeddings (this may take a moment)...")
    generate_embeddings(video_id, frames)
    
    # 5. Build FAISS Index
    print("Building FAISS index...")
    build_index(video_id)
    
    # 6. Search
    query = "a colorful scene"
    print(f"Searching for: '{query}'")
    results = search_by_text(video_id, query, k=3)
    print(f"Top results: {results}")
    
    # 7. Localize Scenes
    print("Localizing scenes...")
    scenes = localize_scenes(video_id, results)
    print(f"Detected scenes: {json.dumps(scenes, indent=2)}")
    
    print("\n--- Pipeline Test Complete ---")

if __name__ == "__main__":
    main()

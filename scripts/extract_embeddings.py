import os
import torch
import json
import numpy as np
from tqdm import tqdm
from src.models.clip_encoder import CLIPEncoder
from src.storage.video_embedding import VideoEmbeddingStorage

def extract_embeddings(video_id, mapping_path, batch_size=32):
    """
    Extracts embeddings for a video given its frame mapping.
    """
    if not os.path.exists(mapping_path):
        print(f"Error: Mapping file not found at {mapping_path}")
        return

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    frame_ids = sorted(list(mapping.keys()))
    frame_paths = [mapping[fid]['path'] for fid in frame_ids]
    
    # Initialize encoder and storage
    encoder = CLIPEncoder()
    storage = VideoEmbeddingStorage()
    
    print(f"Extracting features for {len(frame_paths)} frames in batches of {batch_size}...")
    
    all_embeddings = []
    for i in tqdm(range(0, len(frame_paths), batch_size)):
        batch_paths = frame_paths[i:i+batch_size]
        batch_embeddings = encoder.encode_batch(batch_paths)
        all_embeddings.append(batch_embeddings)
        
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Save to HDF5
    storage.save_embeddings(video_id, all_embeddings)
    print(f"Successfully extracted and saved embeddings for {video_id}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings from video frames.")
    parser.add_argument("--video_id", type=str, required=True, help="ID of the video.")
    parser.add_argument("--mapping", type=str, required=True, help="Path to the frame mapping JSON.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for extraction.")
    
    args = parser.parse_args()
    extract_embeddings(args.video_id, args.mapping, args.batch)

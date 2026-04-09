import os
import json
import requests
import pandas as pd
from tqdm import tqdm
import sys
import torch
import numpy as np

# Fix for OMP: Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Setup paths and imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from src.utils.video_processor import VideoProcessor
from src.retrieval.search import SearchEngine
from src.models.clip_encoder import CLIPEncoder

# Configuration
DATA_DIR = os.path.join(ROOT_DIR, "data/activitynet")

# CURATED STABLE DATASET (Using 100% active, long-form benchmark videos)
CURATED_DATA = {
    "aqz-KE-bpKQ": {
        "title": "Big Buck Bunny (10m)",
        "duration": 600.0,
        "timestamps": [[5.0, 50.0], [60.0, 120.0], [180.0, 300.0]],
        "sentences": ["a large white bunny emerges from a cave and stretches", "the bunny watches a butterfly flying around", "three small animals are plotting in the forest"]
    },
    "dQw4w9WgXcQ": {
        "title": "Rick Astley (3m)",
        "duration": 212.0,
        "timestamps": [[0.0, 25.0], [42.0, 85.0]],
        "sentences": ["a man in a trench coat is singing and dancing", "man is dancing on a stage with a fence background"]
    },
    "9bZkp7q19f0": {
        "title": "Gangnam Style (4m)",
        "duration": 252.0,
        "timestamps": [[5.0, 35.0], [60.0, 100.0]],
        "sentences": ["a man is lounging on a beach chair", "a man is dancing with a crowd in an underground parking lot"]
    }
}

def calculate_iou(pred, gt):
    s1, e1 = pred
    s2, e2 = gt
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - intersection
    return intersection / union if union > 0 else 0

def download_video(video_id, save_path):
    import subprocess
    url = f"https://www.youtube.com/watch?v={video_id}"
    # Added --no-check-certificate and --prefer-free-formats for stability
    cmd = ["yt-dlp", "-f", "best[height<=360]", "--no-check-certificate", "-o", save_path, url]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except:
        return False

def run_benchmark():
    os.makedirs(DATA_DIR, exist_ok=True)
    video_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)

    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"📥 Initializing AI Engines on {device}...")
    clip_encoder = CLIPEncoder(device=device)
    processor = VideoProcessor(device=device, clip_encoder=clip_encoder)
    
    # 1. Download and Index
    print(f"\n🏗️  Step 1: Processing {len(CURATED_DATA)} Stable Long Clips...")
    for vid_id, info in CURATED_DATA.items():
        save_path = os.path.join(video_dir, f"{vid_id}.mp4")
        if not os.path.exists(save_path):
            print(f"🎬 Downloading '{info['title']}'...")
            if not download_video(vid_id, save_path):
                print(f"  ❌ Download failed for {vid_id}. Skipping.")
                continue
        
        processor.process_new_video(save_path, vid_id)

    # 2. Evaluate
    print("\n🔍 Step 2: Evaluating Jump-to-Scene Accuracy (Netflix-style)...")
    engine = SearchEngine(clip_encoder=clip_encoder)
    results_log = []
    
    # SUCCESS WINDOW: How many seconds away counts as "Nearby"?
    NEARBY_THRESHOLD = 10.0 

    for vid_id, info in CURATED_DATA.items():
        if not os.path.exists(os.path.join(video_dir, f"{vid_id}.mp4")): continue
            
        for gt_segment, sentence in zip(info['timestamps'], info['sentences']):
            search_results = engine.search(sentence, top_k=5)
            
            in_scene = False
            near_hit = False
            found_vid = False
            
            gt_start, gt_end = gt_segment

            for res in search_results:
                if res['video_id'] == vid_id:
                    found_vid = True
                    pred_start, pred_end = res['timestamps'][0]
                    
                    # 1. Check if we landed INSIDE the scene
                    if (gt_start <= pred_start <= gt_end) or (gt_start <= pred_end <= gt_end):
                        in_scene = True
                        near_hit = True
                        break
                    
                    # 2. Check if we are within 10 seconds of the boundaries
                    dist_to_start = min(abs(pred_start - gt_start), abs(pred_start - gt_end))
                    if dist_to_start <= NEARBY_THRESHOLD:
                        near_hit = True
            
            results_log.append({
                "found_vid": found_vid,
                "in_scene": in_scene,
                "near_hit": near_hit
            })

    if not results_log:
        print("❌ No evaluation results collected.")
        return

    # 3. Report
    total = len(results_log)
    vid_rec  = sum(1 for r in results_log if r['found_vid']) / total * 100
    in_acc   = sum(1 for r in results_log if r['in_scene']) / total * 100
    near_acc = sum(1 for r in results_log if r['near_hit']) / total * 100

    print("\n" + "📺" * 45)
    print("      NETFLIX-STYLE 'JUMP TO SCENE' BENCHMARK")
    print("📺" * 45)
    print(f"Videos Processed:     {len(CURATED_DATA)}")
    print(f"Queries Run:          {total}")
    print("-" * 45)
    print(f"Video Retrieval:      {vid_rec:.2f}% (Found the right movie)")
    print("-" * 45)
    print(f"In-Scene Accuracy:    {in_acc:.2f}% (Landed directly in the clip)")
    print(f"Nearby Accuracy:      {near_acc:.2f}% (Landed within ±10s of clip)")
    print("-" * 45)
    print("✅ Benchmark Finished.")
    print("📺" * 45 + "\n")

if __name__ == "__main__":
    run_benchmark()

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
    # Use python -m yt_dlp for absolute path safety in venv
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "best[height<=360]",
        "--no-check-certificate",
        "--socket-timeout", "5",
        "--retries", "1",
        "-o", save_path,
        url
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except:
        return False

def setup_data(target_duration_minutes=120):
    os.makedirs(DATA_DIR, exist_ok=True)
    video_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    meta_path = os.path.join(DATA_DIR, "activitynet_v1-3.json")
    
    # Clean up empty or broken metadata files from previous run
    if os.path.exists(meta_path) and os.path.getsize(meta_path) < 1000:
        os.remove(meta_path)
        
    if not os.path.exists(meta_path):
        print("📥 Downloading ActivityNet Metadata (Official Mirror)...")
        # Fixed URL (found inside Evaluation/data/ directory)
        url = "https://raw.githubusercontent.com/activitynet/ActivityNet/master/Evaluation/data/activity_net.v1-3.min.json"
        
        try:
            r = requests.get(url)
            r.raise_for_status()
            raw_data = r.json()
            if isinstance(raw_data, list):
                data = {v['id']: v for v in raw_data}
            elif 'database' in raw_data:
                data = raw_data['database']
            else:
                data = raw_data
            
            with open(meta_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"❌ Failed to download or parse metadata: {e}")
            sys.exit(1)
    else:
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("❌ Existing metadata file is corrupted. Deleting and re-running.")
            os.remove(meta_path)
            return setup_data(target_duration_minutes)

    print(f"📡 Gathering {target_duration_minutes} minutes of clean video...")
    
    # Filter to validation set
    val_set = {k: v for k, v in data.items() if v.get("subset") == "validation"}
    
    accumulated_duration = 0.0
    target_seconds = target_duration_minutes * 60
    selected_videos = {}
    
    # Check existing videos first
    for vid_id, info in val_set.items():
        save_path = os.path.join(video_dir, f"{vid_id}.mp4")
        if os.path.exists(save_path):
            duration = info.get("duration", 180.0)
            accumulated_duration += duration
            selected_videos[vid_id] = info
    
    # Download more if needed
    for vid_id, info in val_set.items():
        if accumulated_duration >= target_seconds:
            break
        if vid_id in selected_videos:
            continue
            
        save_path = os.path.join(video_dir, f"{vid_id}.mp4")
        print(f"  🎬 Trying to download {vid_id}...")
        
        if download_video(vid_id, save_path):
            duration = info.get("duration", 180.0)
            accumulated_duration += duration
            selected_videos[vid_id] = info
            print(f"  ✅ Success. Total gathered: {accumulated_duration/60:.1f} / {target_duration_minutes} minutes.")
        else:
            print(f"  ❌ Failed (Video Removed). Skipping.")

    print(f"\n✅ Ready! Gathered {len(selected_videos)} videos ({accumulated_duration/60:.1f} minutes).")
    return selected_videos

def run_benchmark():
    selected_data = setup_data(target_duration_minutes=120)
    video_dir = os.path.join(DATA_DIR, "videos")

    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"\n📥 Initializing AI Engines on {device}...")
    clip_encoder = CLIPEncoder(device=device)
    processor = VideoProcessor(device=device, clip_encoder=clip_encoder)
    
    # 1. Download and Index
    print(f"\n🏗️  Step 1: Processing {len(selected_data)} Stable Long Clips...")
    for vid_id, info in selected_data.items():
        save_path = os.path.join(video_dir, f"{vid_id}.mp4")
        processor.process_new_video(save_path, vid_id)

    # 2. Evaluate
    print("\n🔍 Step 2: Evaluating Jump-to-Scene Accuracy (Netflix-style)...")
    engine = SearchEngine(clip_encoder=clip_encoder)
    results_log = []
    
    # SUCCESS WINDOW: How many seconds away counts as "Nearby"?
    NEARBY_THRESHOLD = 10.0 

    for vid_id, info in selected_data.items():
        if not os.path.exists(os.path.join(video_dir, f"{vid_id}.mp4")): continue
            
        # Parse official ActivityNet annotations format
        annotations = info.get("annotations", [])
        if not annotations:
            continue
            
        for ann in annotations:
            gt_segment = ann["segment"]
            sentence = ann["label"]  # Using the general label as query, or if you have captions, use them.
            if not sentence: continue

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
    print(f"Videos Processed:     {len(selected_data)}")
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

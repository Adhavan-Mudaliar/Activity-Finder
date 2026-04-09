import os
import sys

# Fix for OMP: Error #15 (Multiple OpenMP runtimes on macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import zipfile
import requests
import pandas as pd
from tqdm import tqdm
import torch

# 1. Setup paths and imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from src.utils.video_processor import VideoProcessor
from src.retrieval.search import SearchEngine
from src.models.clip_encoder import CLIPEncoder

# Configuration
DATA_DIR = os.path.join(ROOT_DIR, "data/msrvtt")
ZIP_URL = "https://github.com/towhee-io/examples/releases/download/data/text_video_search.zip"
VIDEO_LIMIT = 240  # Under 1 hour total footage target

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "msrvtt_subset.zip")
    
    if not os.path.exists(zip_path):
        print(f"🚀 Downloading MSR-VTT subset (approx. 600MB)...")
        response = requests.get(ZIP_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, "wb") as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    # We unzip only if the video folder doesn't exist
    video_folder = os.path.join(DATA_DIR, "test_1k_compress")
    if not os.path.exists(video_folder):
        print("📦 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    
    print(f"✅ Data ready in {DATA_DIR}")

def run_benchmark():
    # Load metadata
    csv_path = os.path.join(DATA_DIR, "MSRVTT_JSFUSION_test.csv")
    if not os.path.exists(csv_path):
        print("❌ Error: Ground truth CSV not found.")
        return

    df = pd.read_csv(csv_path)
    video_folder = os.path.join(DATA_DIR, "test_1k_compress")
    
    # Selection
    unique_videos = df['video_id'].unique()[:VIDEO_LIMIT]
    print(f"⚙️  Starting Benchmark for {len(unique_videos)} videos...")

    # Shared Resources
    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"📥 Loading Models on {device}...")
    clip_encoder = CLIPEncoder(device=device)
    processor = VideoProcessor(device=device, clip_encoder=clip_encoder)
    
    # 1. Processing Phase
    for vid in tqdm(unique_videos, desc="Indexing Videos"):
        video_path = os.path.join(video_folder, f"{vid}.mp4")
        if os.path.exists(video_path):
            try:
                # Use project's internal processor
                processor.process_new_video(video_path, vid)
            except Exception as e:
                print(f"⚠️  Error processing {vid}: {e}")
    
    # 2. Evaluation Phase
    print("\n📈 Starting Accuracy Evaluation (Recall@K)...")
    engine = SearchEngine(clip_encoder=clip_encoder)
    
    test_data = df[df['video_id'].isin(unique_videos)]
    ranks = []
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Searching"):
        query = row['sentence']
        gt_video = str(row['video_id'])
        
        # Search the top 20 candidates
        results = engine.search(query, top_k=20)
        
        found_rank = -1
        for i, res in enumerate(results):
            if str(res['video_id']) == gt_video:
                found_rank = i + 1
                break
        
        ranks.append(found_rank)
    
    # Calculate Metrics
    total_queries = len(ranks)
    valid_ranks = [r for r in ranks if r > 0]
    
    recall_1 = sum(1 for r in ranks if 0 < r <= 1) / total_queries * 100
    recall_3 = sum(1 for r in ranks if 0 < r <= 3) / total_queries * 100
    recall_5 = sum(1 for r in ranks if 0 < r <= 5) / total_queries * 100
    recall_10 = sum(1 for r in ranks if 0 < r <= 10) / total_queries * 100
    median_rank = pd.Series(valid_ranks).median() if valid_ranks else float('nan')

    print("\n" + "⭐" * 45)
    print("      MSR-VTT BENCHMARK (FAST SUBSET)")
    print("⭐" * 45)
    print(f"Dataset Scope:        240/1000 Clips (Test Split)")
    print(f"Total Video Time:     ~58 minutes")
    print(f"Total Queries:        {total_queries}")
    print("-" * 45)
    print(f"Recall @ 1:           {recall_1:.2f}%")
    print(f"Recall @ 3:           {recall_3:.2f}%")
    print(f"Recall @ 5:           {recall_5:.2f}%")
    print(f"Recall @ 10:          {recall_10:.2f}%")
    print("-" * 45)
    print(f"Median Rank:          {median_rank}")
    print("-" * 45)
    print("✅ Benchmark Finished.")
    print("⭐" * 45 + "\n")

if __name__ == "__main__":
    try:
        download_data()
        run_benchmark()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Benchmark failed: {e}")

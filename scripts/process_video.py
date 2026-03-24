import os
import cv2
import argparse
from src.sampling.hybrid_sampler import HybridSampler
from src.utils.metadata_manager import MetadataManager

def process_video(video_path, uniform_interval=30):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    video_id = os.path.basename(video_path).split('.')[0]
    output_frames_dir = os.path.join("data/frames", video_id)
    
    # Initialize sampler and metadata manager
    sampler = HybridSampler(uniform_interval=uniform_interval)
    metadata_manager = MetadataManager()

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    # Sample frames
    print(f"Sampling frames for {video_id}...")
    frame_mapping = sampler.sample_frames(video_path, output_frames_dir)

    # Save metadata
    metadata_manager.add_video_metadata(video_id, video_path, duration)
    metadata_manager.save_frame_mapping(video_id, frame_mapping)

    print(f"Successfully processed {video_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for scene retrieval project.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--interval", type=int, default=30, help="Uniform sampling interval.")
    
    args = parser.parse_args()
    process_video(args.video, args.interval)

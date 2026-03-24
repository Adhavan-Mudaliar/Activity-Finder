import cv2
import os
import numpy as np
from src.sampling.scene_detector import SceneDetector

class HybridSampler:
    def __init__(self, uniform_interval=30, scene_threshold=30.0):
        """
        Args:
            uniform_interval (int): Every N frames to sample uniformly.
            scene_threshold (float): Threshold for ContentDetector in SceneDetect.
        """
        self.uniform_interval = uniform_interval
        self.detector = SceneDetector(threshold=scene_threshold)

    def sample_frames(self, video_path, output_dir, video_id=None):
        """
        Extracts frames using both uniform and scene detection methods.
        Stores them in output_dir and returns a mapping of frame_id -> timestamp.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_id is None:
            video_id = os.path.splitext(os.path.basename(video_path))[0]

        # 1. Uniform Sampling
        uniform_frames = list(range(0, total_frames, self.uniform_interval))

        # 2. Scene Detection
        scenes = self.detector.detect_scenes(video_path)
        scene_frames = self.detector.get_scene_frames(video_path, scenes)

        # 3. Combine and Sort (ensure unique)
        all_frames = sorted(list(set(uniform_frames + scene_frames)))

        frame_mapping = {}
        for frame_idx in all_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_idx / fps
            frame_id = f"{video_id}_frame_{frame_idx:06d}"
            frame_path = os.path.join(output_dir, f"{frame_id}.jpg")
            
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"⚠️  Warning: Failed to save frame {frame_path}")
                continue
                
            frame_mapping[frame_id] = {
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "path": frame_path
            }

        cap.release()
        print(f"Extracted {len(frame_mapping)} frames from {video_path}.")
        return frame_mapping

if __name__ == "__main__":
    # Test
    # sampler = HybridSampler(uniform_interval=50)
    # mapping = sampler.sample_frames("data/raw_videos/sample.mp4", "data/frames/sample")
    # print(mapping)
    pass

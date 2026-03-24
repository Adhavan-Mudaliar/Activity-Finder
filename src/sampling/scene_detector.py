import cv2
import os
import numpy as np

class SceneDetector:
    def __init__(self, threshold=0.3, min_scene_len=15):
        """
        Threshold for histogram correlation (0 to 1, higher means more stable).
        We use (1 - correlation) > threshold to detect scene change.
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect_scenes(self, video_path):
        """
        Detects scenes in a video using histogram difference (fallback).
        Returns a list of (start_time, end_time) tuples.
        """
        print(f"Detecting scenes (fallback) for {video_path}...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        last_hist = None
        scene_start_frame = 0
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Compute histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            
            if last_hist is not None:
                # Compare histograms
                correlation = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                if (1.0 - correlation) > self.threshold:
                    if (frame_idx - scene_start_frame) >= self.min_scene_len:
                        scenes.append((scene_start_frame / fps, frame_idx / fps))
                        scene_start_frame = frame_idx
            
            last_hist = hist
            
        # Add last scene
        if (total_frames - scene_start_frame) > 0:
            scenes.append((scene_start_frame / fps, total_frames / fps))
            
        cap.release()
        print(f"Found {len(scenes)} scenes (fallback).")
        return scenes

    def get_scene_frames(self, video_path, scenes):
        """
        Extracts the center frame from each detected scene.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        scene_frames = []
        for start, end in scenes:
            mid_time = (start + end) / 2
            frame_num = int(mid_time * fps)
            scene_frames.append(frame_num)
            
        cap.release()
        return scene_frames

if __name__ == "__main__":
    # Test fallback
    detector = SceneDetector()
    # scenes = detector.detect_scenes("data/videos/testX.mp4")
    # print(scenes)

"""
Frame-level index metadata store.
Maps FAISS integer index → (video_id, frame_idx, timestamp).
Persisted alongside the FAISS index so searches can recover timestamps directly.
"""
import json
import os

class FrameIndexStore:
    def __init__(self):
        # List of dicts: {"video_id": ..., "frame_idx": ..., "timestamp": ...}
        self.records: list[dict] = []

    def add(self, video_id: str, frame_idx: int, timestamp: float):
        self.records.append({"video_id": video_id, "frame_idx": frame_idx, "timestamp": timestamp})

    def get(self, faiss_idx: int) -> dict | None:
        if 0 <= faiss_idx < len(self.records):
            return self.records[faiss_idx]
        return None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f)

    def load(self, path: str):
        with open(path, "r") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

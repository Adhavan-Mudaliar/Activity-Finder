import os
import csv
import json

class MetadataManager:
    def __init__(self, metadata_dir="data/metadata"):
        self.metadata_dir = metadata_dir
        self.metadata_file = os.path.join(metadata_dir, "metadata.csv")
        self.frame_mapping_dir = os.path.join(metadata_dir, "frame_mappings")
        
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
        if not os.path.exists(self.frame_mapping_dir):
            os.makedirs(self.frame_mapping_dir)

    def add_video_metadata(self, video_id, path, duration, captions=""):
        """
        Adds video metadata to metadata.csv.
        """
        file_exists = os.path.isfile(self.metadata_file)
        with open(self.metadata_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['video_id', 'path', 'duration', 'captions'])
            writer.writerow([video_id, path, duration, captions])
        print(f"Added metadata for video: {video_id}")

    def save_frame_mapping(self, video_id, frame_mapping):
        """
        Saves frame_id -> timestamp mapping for a specific video.
        """
        mapping_path = os.path.join(self.frame_mapping_dir, f"{video_id}_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(frame_mapping, f, indent=4)
        print(f"Saved frame mapping for {video_id} to {mapping_path}")

    def get_video_metadata(self, video_id):
        """
        Retrieves metadata for a specific video.
        """
        if not os.path.isfile(self.metadata_file):
            return None
        with open(self.metadata_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['video_id'] == video_id:
                    return row
        return None

if __name__ == "__main__":
    # Test
    # manager = MetadataManager()
    # manager.add_video_metadata("sample", "data/raw_videos/sample.mp4", 120, "Sample video for testing")
    pass

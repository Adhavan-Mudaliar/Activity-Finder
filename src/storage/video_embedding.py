import h5py
import numpy as np
import os

class VideoEmbeddingStorage:
    def __init__(self, h5_path="data/hdf5/embeddings.h5"):
        """
        Initializes the HDF5 storage for video embeddings.
        """
        self.h5_path = h5_path
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    def save_embeddings(self, video_id, embeddings):
        """
        Saves video embeddings to HDF5.
        embeddings: (num_frames, dimension) numpy array.
        """
        with h5py.File(self.h5_path, 'a') as f:
            if video_id in f:
                print(f"Dataset for {video_id} already exists. Overwriting...")
                del f[video_id]
            
            f.create_dataset(video_id, data=embeddings, compression="gzip")
        print(f"Saved {embeddings.shape[0]} embeddings for {video_id} to {self.h5_path}.")

    def get_embeddings(self, video_id):
        """
        Retrieves embeddings for a specific video.
        """
        with h5py.File(self.h5_path, 'r') as f:
            if video_id not in f:
                print(f"No embeddings found for {video_id}.")
                return None
            return np.array(f[video_id])

    def list_videos(self):
        """
        Lists all video IDs in the HDF5 file.
        """
        if not os.path.exists(self.h5_path):
            return []
        with h5py.File(self.h5_path, 'r') as f:
            return list(f.keys())

if __name__ == "__main__":
    # Test
    # storage = VideoEmbeddingStorage()
    # storage.save_embeddings("test_video", np.random.rand(10, 512))
    # print(storage.get_embeddings("test_video").shape)
    pass

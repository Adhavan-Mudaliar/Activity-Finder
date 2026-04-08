import torch
import numpy as np
import os
import unittest
from src.storage.video_embedding import VideoEmbeddingStorage

class TestComponents(unittest.TestCase):
    def test_storage(self):
        h5_path = "data/hdf5/test_storage.h5"
        if os.path.exists(h5_path): os.remove(h5_path)
        
        storage = VideoEmbeddingStorage(h5_path=h5_path)
        data = np.random.rand(5, 512)
        storage.save_embeddings("test_vid", data)
        
        loaded_data = storage.get_embeddings("test_vid")
        np.testing.assert_array_almost_equal(data, loaded_data)
        
        if os.path.exists(h5_path): os.remove(h5_path)

if __name__ == "__main__":
    unittest.main()


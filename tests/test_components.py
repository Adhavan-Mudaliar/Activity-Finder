import torch
import numpy as np
import os
import unittest
from src.models.projection_head import ProjectionHead
from src.models.mamba_model import MambaModel
from src.models.attention_pooling import AttentionPooling
from src.storage.video_embedding import VideoEmbeddingStorage

class TestComponents(unittest.TestCase):
    def test_projection_head(self):
        model = ProjectionHead(input_dim=512, output_dim=256)
        x = torch.randn(2, 10, 512)
        y = model(x)
        self.assertEqual(y.shape, (2, 10, 256))

    def test_mamba_model(self):
        model = MambaModel(d_model=256, n_layers=2)
        x = torch.randn(2, 10, 256)
        y = model(x)
        self.assertEqual(y.shape, (2, 10, 256))

    def test_attention_pooling(self):
        model = AttentionPooling(d_model=256)
        x = torch.randn(2, 10, 256)
        y, attn = model(x)
        self.assertEqual(y.shape, (2, 256))
        self.assertEqual(attn.shape, (2, 10))

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

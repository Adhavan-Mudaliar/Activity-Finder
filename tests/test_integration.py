import unittest
import torch
import os
from src.retrieval.search import SearchEngine

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # We assume data/metadata/faiss exists from our previous build_index run
        self.engine = SearchEngine()
        self.engine.faiss_index.load("data/metadata/faiss")

    def test_full_search_flow(self):
        query = "a futuristic city with a car"
        print(f"Testing integration for query: {query}")
        
        results = self.engine.search(query, top_k=5)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for res in results:
            self.assertIn("video_id", res)
            self.assertIn("score", res)
            self.assertIn("timestamps", res)
            print(f"Found match: {res['video_id']} with score {res['score']:.4f}")
            print(f"Timestamps: {res['timestamps']}")

if __name__ == "__main__":
    unittest.main()

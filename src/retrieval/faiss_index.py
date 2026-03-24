import faiss
import numpy as np
import os

class FaissIndex:
    def __init__(self, dimension=256, index_type="IVF", nlist=1):
        """
        Initializes FAISS index for scene embeddings.
        Decreased default nlist to 1 for small datasets.
        """
        self.dimension = dimension
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dimension)
            # Ensure nlist is not larger than data size if known, 
            # but usually we don't know yet. Setting to 1 as safe default.
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dimension)
            
        self.video_ids = [] # To map index to video_id

    def add_embeddings(self, embeddings, video_id):
        """
        Adds embeddings for a video to the index.
        embeddings: (1, dimension) or (num_scenes, dimension)
        """
        if not self.index.is_trained:
            self.index.train(embeddings.astype('float32'))
            
        self.index.add(embeddings.astype('float32'))
        for _ in range(embeddings.shape[0]):
            self.video_ids.append(video_id)

    def search(self, query_embedding, top_k=100, video_id=None):
        """
        Searches the index for the query.
        """
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            res_vid = self.video_ids[idx]
            if video_id and res_vid != video_id:
                continue
                
            results.append({
                "video_id": res_vid,
                "score": float(distances[0][i])
            })
        return results

    def save(self, path):
        faiss.write_index(self.index, path + ".index")
        with open(path + "_ids.txt", "w") as f:
            for vid in self.video_ids:
                f.write(vid + "\n")

    def load(self, path):
        self.index = faiss.read_index(path + ".index")
        with open(path + "_ids.txt", "r") as f:
            self.video_ids = [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    # Test
    # index = FaissIndex(dimension=256)
    # data = np.random.rand(100, 256).astype('float32')
    # index.add_embeddings(data, "test")
    # res = index.search(np.random.rand(1, 256).astype('float32'))
    # print(res[:5])
    pass

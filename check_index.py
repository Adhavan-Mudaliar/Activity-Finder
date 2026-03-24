import faiss
import os

index_path = "data/metadata/faiss"
if os.path.exists(index_path + ".index"):
    index = faiss.read_index(index_path + ".index")
    print(f"Index dimension: {index.d}")
    print(f"Total vectors: {index.ntotal}")
    print(f"Is trained: {index.is_trained}")
    
    with open(index_path + "_ids.txt", "r") as f:
        video_ids = [line.strip() for line in f.readlines()]
    print(f"Video IDs mapping length: {len(video_ids)}")
    print(f"Video IDs: {video_ids}")
    
    # Try a dummy search with a random vector
    import numpy as np
    query = np.random.rand(1, index.d).astype('float32')
    distances, indices = index.search(query, 100)
    print(f"Search results for random query: indices={indices[0][:5]}, distances={distances[0][:5]}")
else:
    print("Index not found.")

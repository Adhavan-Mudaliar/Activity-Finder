import numpy as np

def create_sliding_windows(embeddings, window_size=10, stride=5):
    """
    Groups frame embeddings into sliding temporal windows.
    embeddings: (N, D) array of frame embeddings
    Returns: (num_windows, window_size, D) array of windowed embeddings.
             If N < window_size, returns (1, N, D).
    """
    num_frames = embeddings.shape[0]
    
    if num_frames <= window_size:
        return np.expand_dims(embeddings, axis=0) # (1, N, D)
        
    windows = []
    for i in range(0, num_frames - window_size + 1, stride):
        windows.append(embeddings[i:i + window_size])
        
    # Ensure the last frames are captured if stride didn't align perfectly
    if (num_frames - window_size) % stride != 0:
        # Don't add a duplicate if the last stride perfectly hit the end
        if i + stride < num_frames - window_size:
             windows.append(embeddings[-window_size:])
             
    if len(windows) == 0:
        return np.expand_dims(embeddings, axis=0)
        
    return np.array(windows)

import numpy as np
from scipy.ndimage import gaussian_filter

class TemporalLocalizer:
    def __init__(self, sigma=2.0, alpha=1.0, window_size=10, stride=5):
        self.sigma = sigma
        self.alpha = alpha
        self.window_size = window_size
        self.stride = stride

    def compute_frame_level_similarity(self, query_embedding, frame_embeddings):
        # (L, D) @ (D, 1) -> (L, 1)
        scores = np.dot(frame_embeddings, query_embedding.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([scores])
        return scores

    def _compute_hybrid_window_scores(self, frame_scores):
        if len(frame_scores) <= self.window_size:
            return np.array([np.mean(frame_scores)])
            
        windows = []
        for i in range(0, len(frame_scores) - self.window_size + 1, self.stride):
            windows.append(frame_scores[i:i + self.window_size])
            
        if (len(frame_scores) - self.window_size) % self.stride != 0:
            windows.append(frame_scores[-self.window_size:])
            
        windows = np.array(windows)
        
        # w1: semantic similarity (mean score)
        sem_sim = np.mean(windows, axis=1)
        
        # w2: temporal coherence (inverse of variance - smaller variance means more coherent)
        variance = np.var(windows, axis=1)
        temp_coh = np.exp(-variance * 10) # Exponential decay to scale to [0,1] roughly
        
        hybrid_scores = 0.6 * sem_sim + 0.2 * temp_coh
        
        # w3: neighbor agreement
        final_scores = np.copy(hybrid_scores)
        for i in range(1, len(hybrid_scores) - 1):
            neighbor_avg = (hybrid_scores[i-1] + hybrid_scores[i+1]) / 2.0
            final_scores[i] += 0.2 * neighbor_avg
            
        return final_scores

    def localize(self, frame_scores, extraction_fps=1.0): # Usually 1 frame per 30 frames (which is ~1 fps)
        if len(frame_scores) == 0:
            return []

        # 1. Hybrid scoring over windows
        hybrid_scores = self._compute_hybrid_window_scores(frame_scores)
        
        # 2. Gaussian smoothing
        smoothed_scores = gaussian_filter(hybrid_scores, sigma=self.sigma)
        
        # 3. Dynamic Threshold
        mean_score = np.mean(smoothed_scores)
        std_score = np.std(smoothed_scores)
        threshold = mean_score + self.alpha * std_score
        
        # 4. Segment Grouping
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, score in enumerate(smoothed_scores):
            if score > threshold:
                if not in_segment:
                    in_segment = True
                    start_idx = i
            else:
                if in_segment:
                    in_segment = False
                    # convert window index to time
                    start_time = (start_idx * self.stride) / extraction_fps
                    end_time = (i * self.stride + self.window_size) / extraction_fps
                    segments.append([start_time, end_time])
                    
        if in_segment:
            start_time = (start_idx * self.stride) / extraction_fps
            end_time = (len(smoothed_scores) * self.stride + self.window_size) / extraction_fps
            segments.append([start_time, end_time])
            
        return segments

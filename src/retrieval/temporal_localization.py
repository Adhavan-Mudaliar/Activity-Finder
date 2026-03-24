import numpy as np
from scipy.ndimage import gaussian_filter

class TemporalLocalizer:
    def __init__(self, sigma=2.0, alpha=1.0):
        """
        sigma: Standard deviation for Gaussian smoothing.
        alpha: Coefficient for dynamic threshold calculation.
        """
        self.sigma = sigma
        self.alpha = alpha

    def compute_frame_level_similarity(self, query_embedding, frame_embeddings):
        """
        query_embedding: (1, D)
        frame_embeddings: (L, D)
        Returns: Scores (L,)
        """
        # (L, D) @ (D, 1) -> (L, 1)
        # Assuming both are normalized
        scores = np.dot(frame_embeddings, query_embedding.T).squeeze()
        return scores

    def localize(self, scores, fps=1.0):
        """
        scores: Frame-level similarity scores (L,)
        fps: Frames per second of extraction.
        Returns: List of [start_time, end_time] segments.
        """
        if len(scores) == 0:
            return []

        # Step 2: Gaussian smoothing
        smoothed_scores = gaussian_filter(scores, sigma=self.sigma)
        
        # Step 3: Dynamic Threshold
        mean_score = np.mean(smoothed_scores)
        std_score = np.std(smoothed_scores)
        threshold = mean_score + self.alpha * std_score
        
        # Step 4: Segment Grouping
        segments = []
        in_segment = False
        start_frame = 0
        
        for i, score in enumerate(smoothed_scores):
            if score > threshold:
                if not in_segment:
                    in_segment = True
                    start_frame = i
            else:
                if in_segment:
                    in_segment = False
                    segments.append([start_frame / fps, i / fps])
                    
        # Add last segment if active
        if in_segment:
            segments.append([start_frame / fps, len(smoothed_scores) / fps])
            
        return segments

if __name__ == "__main__":
    # Test
    # localizer = TemporalLocalizer()
    # scores = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 0.8, 0.4, 0.2, 0.1])
    # segs = localizer.localize(scores)
    # print(segs)
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

class RankingNetwork(nn.Module):
    def __init__(self, query_dim=256, scene_dim=256, hidden_dim=128):
        """
        Ranking Network: Input [query, scene].
        2-layer MLP to output a relevance score.
        """
        super(RankingNetwork, self).__init__()
        self.fc1 = nn.Linear(query_dim + scene_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, query, scene):
        """
        query: (B, D)
        scene: (B, D)
        """
        # Ensure correct shapes
        if query.dim() == 3:
            query = query.squeeze(1)
        if scene.dim() == 3:
            scene = scene.squeeze(1)
            
        x = torch.cat([query, scene], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # We output raw logits for BCEWithLogitsLoss or sigmoid for score
        return x

if __name__ == "__main__":
    # Test
    model = RankingNetwork()
    q = torch.randn(8, 256)
    s = torch.randn(8, 256)
    y = model(q, s)
    print(y.shape) # (8, 1)

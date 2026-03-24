import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, d_model=256):
        """
        Attention Pooling to condense sequence into a scene embedding.
        """
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        x: (B, L, D)
        mask: (B, L) with 1s for valid frames, 0s for padding.
        """
        attn_weights = self.attention(x).squeeze(-1) # (B, L)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=-1) # (B, L)
        
        # Weighted sum: (B, 1, L) @ (B, L, D) -> (B, 1, D)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return pooled, attn_weights

if __name__ == "__main__":
    # Test
    model = AttentionPooling()
    x = torch.randn(8, 20, 256)
    y, attn = model(x)
    print(y.shape) # (8, 256)

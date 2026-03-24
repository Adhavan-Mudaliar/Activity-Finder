import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        """
        Cross Attention: Query attends to frame sequence.
        """
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):
        """
        query: (B, 1, D) - Text query or scene summary.
        key_value: (B, L, D) - Frame sequence.
        mask: (B, L) - 1s for valid frames, 0s for padding.
        """
        # MultiheadAttention expects key_padding_mask as (B, L) with True for padded
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        attn_out, attn_weights = self.multihead_attn(
            query, key_value, key_value, 
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection and norm
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights

if __name__ == "__main__":
    # Test
    model = CrossAttention()
    query = torch.randn(8, 1, 256)
    kv = torch.randn(8, 20, 256)
    y, attn = model(query, kv)
    print(y.shape) # (8, 1, 256)

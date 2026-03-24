import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=256, dropout=0.1):
        """
        Residual MLP Projection Head.
        512 -> 256
        """
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.LayerNorm(output_dim)
        
        self.proj_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual = self.proj_residual(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

if __name__ == "__main__":
    # Test
    model = ProjectionHead()
    x = torch.randn(8, 10, 512)
    y = model(x)
    print(y.shape) # Should be (8, 10, 256)

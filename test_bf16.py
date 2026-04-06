import torch
import numpy as np

try:
    x = torch.randn(2, 2, dtype=torch.bfloat16, device="mps")
    x /= x.norm(dim=-1, keepdim=True)
    print("Norm worked!")
    x.cpu().numpy()
    print("Numpy worked!")
except Exception as e:
    print("Error:", e)


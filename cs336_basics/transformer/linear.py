import torch
import torch.nn as nn
import math
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean = 0
        self.std_sq = 2/(self.in_features+self.out_features)
        self.std = math.sqrt(self.std_sq)
        self.W = nn.Parameter(torch.empty(
            (out_features, in_features), device=device, dtype=dtype))
        # initialize the weights
        torch.nn.init.trunc_normal_(
            self.W, self.mean, self.std, -3*self.std, 3*self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.W, " ... d_in ,d_out d_in-> ... d_out")
        return y

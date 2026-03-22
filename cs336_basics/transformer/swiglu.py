import torch
import torch.nn as nn
from einops import einsum
from cs336_basics.transformer.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int,
                 d_ff: int | None = None,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model

        if d_ff is not None:
            self.d_ff = d_ff
        else:
            self.d_ff = int((8/3)*self.d_model)

        self.w1_weight = Linear(self.d_model, self.d_ff,
                                device=device, dtype=dtype)
        self.w2_weight = Linear(self.d_ff, self.d_model,
                                device=device, dtype=dtype)
        self.w3_weight = Linear(self.d_model, self.d_ff,
                                device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m1 = self.w1_weight(x)
        m2 = self.w3_weight(x)
        m3 = self.silu(m1) * (m2)
        return self.w2_weight(m3)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x/(1+torch.exp(-x))

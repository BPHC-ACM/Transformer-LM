import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.device = device
        self.gain = nn.Parameter(torch.empty(
            (self.d_model, ), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # all math is element wise here
        in_dtype = x.dtype
        x = x.to(torch.float32)
        denom = torch.sqrt((torch.sum(torch.square(x), dim=2,
                                      keepdim=True)/self.d_model)+self.eps)
        result = (x/denom)*self.gain
        return result.to(in_dtype)

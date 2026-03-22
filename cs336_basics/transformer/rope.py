import torch
import torch.nn as nn
from einops import einsum


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # 1. Calculate the inverse frequencies
        # We step by 2 because each frequency applies to a pair of values
        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (k / d_k))

        # 2. Create the position indices (0 to max_seq_len - 1)
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32)

        # 3. Outer product to get all angles (i * theta)
        # shape: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, inv_freq)

        # 4. Duplicate each angle so it applies to both the x and y coordinates of the pair
        # shape becomes: (max_seq_len, d_k)
        angles = angles.repeat_interleave(2, dim=-1)

        # 5. Precompute cos and sin and register as non-persistent buffers
        # persistent=False means they won't be saved in the model's state_dict checkpoint
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Advanced indexing: this looks up the angles and automatically
        # reshapes them to (..., seq_len, d_k) to match x!
        # We also safely cast them to match x's dtype (e.g. if x is float16 or bfloat16)
        cos = self.cos_cached[token_positions].to(x.dtype)
        sin = self.sin_cached[token_positions].to(x.dtype)

        # The rotation trick to avoid the sparse matrix:
        # x[..., 0::2] grabs the evens (x0, x2, x4...)
        # x[..., 1::2] grabs the odds (x1, x3, x5...)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Stack them as [-x2, x1] and flatten back out
        # This transforms [x0, x1, x2, x3] into [-x1, x0, -x3, x2]
        x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2, -1)

        # Apply the final element-wise rotation formula (Equation 8)
        return (x * cos) + (x_rotated * sin)

import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()

        # multi-head attention sublayer
        self.norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype
        )

        # swiglu feed-forward sublayer
        self.norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model=d_model, d_ff=d_ff,
                         device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # first residual stream for attention
        x = x + self.mha.forward_with_rope(self.norm_1(x), token_positions)

        # second residual stream for feed forward
        x = x + self.ff(self.norm_2(x))

        return x

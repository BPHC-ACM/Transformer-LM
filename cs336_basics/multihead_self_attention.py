import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RoPE
from cs336_basics.linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048, rope_theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # separate layers so the adapter can inject weights directly
        self.W_q = Linear(d_model, d_model,
                             device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model,
                             device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model,
                             device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model,
                             device=device, dtype=dtype)

        # rope module
        self.rope = RoPE(theta=rope_theta, d_k=self.head_dim,
                         max_seq_len=max_seq_len, device=device)

    def forward_without_rope(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)

        # project and split heads
        q = rearrange(self.W_q(x), "... s (h d) -> ... h s d",
                      h=self.num_heads)
        k = rearrange(self.W_k(x), "... s (h d) -> ... h s d",
                      h=self.num_heads)
        v = rearrange(self.W_v(x), "... s (h d) -> ... h s d",
                      h=self.num_heads)

        # causal mask
        mask = torch.tril(torch.ones((seq_len, seq_len),
                          dtype=torch.bool, device=x.device))

        # attention
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # concat and project
        out = rearrange(out, "... h s d -> ... s (h d)")
        return self.W_o(out)

    def forward_with_rope(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)

        # project and split heads
        q = rearrange(self.W_q(x), "... s (h d) -> ... h s d",
                      h=self.num_heads)
        k = rearrange(self.W_k(x), "... s (h d) -> ... h s d",
                      h=self.num_heads)
        v = rearrange(self.W_v(x), "... s (h d) -> ... h s d",
                      h=self.num_heads)

        # apply rope (unsqueeze to broadcast across num_heads)
        token_positions_bc = token_positions.unsqueeze(-2)
        q = self.rope(q, token_positions_bc)
        k = self.rope(k, token_positions_bc)

        # causal mask
        mask = torch.tril(torch.ones((seq_len, seq_len),
                          dtype=torch.bool, device=x.device))

        # attention
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # concat and project
        out = rearrange(out, "... h s d -> ... s (h d)")
        return self.W_o(out)

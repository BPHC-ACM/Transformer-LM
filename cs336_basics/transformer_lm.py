import torch
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()

        # token embeddings only (rope handles positions)
        self.token_embed = nn.Embedding(
            vocab_size, d_model, device=device, dtype=dtype)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        # final norm and lm head
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape

        # embed tokens
        x = self.token_embed(x)

        # token positions are still needed to pass into the blocks for rope
        positions = torch.arange(
            seq_len, device=x.device).unsqueeze(0).expand(b, seq_len)

        for block in self.blocks:
            x = block(x, positions)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

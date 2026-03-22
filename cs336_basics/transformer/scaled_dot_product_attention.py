import math
import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:

    d_k = q.size(-1)
    scores = torch.einsum("... i d, ... j d -> ... i j", q, k)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    attn_probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("... i j, ... j v -> ... i v", attn_probs, v)

    return output

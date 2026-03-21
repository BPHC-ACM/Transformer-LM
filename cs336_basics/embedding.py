import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.mean = 0
        self.std = 1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Parameter(torch.empty(
            (self.num_embeddings, self.embedding_dim), device=device, dtype=dtype))
        # initialize the embedding matrix
        torch.nn.init.trunc_normal_(
            self.embedding, self.mean, self.std, -3*self.std, 3*self.std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]

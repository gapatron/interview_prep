"""Solution: Multi-head attention — project Q,K,V, reshape to heads, SDPA, concat, project."""

import math
import torch
import torch.nn as nn
from solution_scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.dropout(self.W_out(out))


if __name__ == "__main__":
    B, L, d_model, H = 2, 4, 8, 2
    mha = MultiHeadAttention(d_model, H)
    x = torch.randn(B, L, d_model)
    out = mha(x)
    assert out.shape == (B, L, d_model)
    print("MultiHeadAttention OK.")

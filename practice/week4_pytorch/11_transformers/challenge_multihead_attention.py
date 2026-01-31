"""
Transformers — Multi-Head Attention Block
=========================================
Build a multi-head self-attention module: project Q, K, V from d_model into num_heads
heads of dimension head_dim = d_model // num_heads, run scaled dot-product attention per head,
concat heads, then project back to d_model.
  - W_q, W_k, W_v: Linear(d_model, d_model); then reshape to (B, num_heads, L, head_dim).
  - Run your scaled_dot_product_attention on each head (or batched).
  - Concat heads -> (B, L, d_model), then W_out: Linear(d_model, d_model).
Return output (B, L, d_model) and optional attn_weights.
"""

import torch
import torch.nn as nn
import math


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
        # x: (B, L, d_model)
        B, L, _ = x.shape
        # TODO: Q = self.W_q(x), K = self.W_k(x), V = self.W_v(x)
        # TODO: Reshape to (B, num_heads, L, head_dim)
        # TODO: Run scaled dot-product attention (per head batched); scale by sqrt(head_dim)
        # TODO: Concat heads -> (B, L, d_model), apply W_out and dropout
        pass


if __name__ == "__main__":
    from solution_scaled_dot_product_attention import scaled_dot_product_attention
    B, L, d_model, H = 2, 4, 8, 2
    mha = MultiHeadAttention(d_model, H)
    x = torch.randn(B, L, d_model)
    out = mha(x)
    assert out.shape == (B, L, d_model)
    print("MultiHeadAttention OK.")

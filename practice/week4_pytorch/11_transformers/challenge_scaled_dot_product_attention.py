"""
11 — Scaled dot-product attention
==================================
Learning goal: Attention(Q,K,V) = softmax(QK^T / √d_k) V. Scale by √d_k so softmax doesn’t saturate; optional mask sets invalid positions to -1e9 before softmax.

Implement:
  - scaled_dot_product_attention(Q, K, V, mask=None): scores = Q @ K^T / sqrt(d_k); if mask: add -1e9 where mask is set; attn = softmax(scores, dim=-1); return (attn @ V, attn).
  - Q,K,V shape (B, L, d_k); output (B, L, d_k), attn_weights (B, L, L).

Run the assert to check shapes and that attn sums to 1 over the last dim. See Advanced_Architectures_Guide (transformers chapter).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V: (B, L, d_k). Return (output (B, L, d_k), attn_weights (B, L, L)).
    If mask is not None, shape (L, L) or (1, 1, L, L); mask value 0 = keep, 1 = mask out (use -1e9).
    """
    # TODO: d_k = Q.size(-1)
    # TODO: scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # TODO: if mask is not None: mask value 1 = mask out (set scores to -1e9)
    # TODO: attn_weights = F.softmax(scores, dim=-1)
    # TODO: output = torch.matmul(attn_weights, V)
    # TODO: return output, attn_weights
    pass


if __name__ == "__main__":
    import math
    B, L, d_k = 2, 4, 8
    Q = torch.randn(B, L, d_k)
    K = torch.randn(B, L, d_k)
    V = torch.randn(B, L, d_k)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, L, d_k) and attn.shape == (B, L, L)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, L))
    print("Scaled dot-product attention OK.")

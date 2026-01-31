"""Solution: Scaled dot-product attention with optional mask."""

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 1, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


if __name__ == "__main__":
    B, L, d_k = 2, 4, 8
    Q = torch.randn(B, L, d_k)
    K = torch.randn(B, L, d_k)
    V = torch.randn(B, L, d_k)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, L, d_k) and attn.shape == (B, L, L)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, L))
    print("Scaled dot-product attention OK.")

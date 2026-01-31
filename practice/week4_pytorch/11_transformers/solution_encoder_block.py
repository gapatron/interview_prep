"""Solution: Encoder block with Pre-LN, self-attention, FFN, residuals."""

import torch
import torch.nn as nn
from solution_multihead_attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


if __name__ == "__main__":
    B, L, d_model, H, d_ff = 2, 4, 8, 2, 32
    block = TransformerEncoderBlock(d_model, H, d_ff)
    x = torch.randn(B, L, d_model)
    out = block(x)
    assert out.shape == (B, L, d_model)
    print("TransformerEncoderBlock OK.")

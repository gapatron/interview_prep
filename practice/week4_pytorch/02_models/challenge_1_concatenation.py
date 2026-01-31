"""
02 — Concatenating features (ConcatBlock)
=========================================
Learning goal: Combine two feature vectors (e.g. from different branches or a skip connection)
by concatenating along the feature dimension and projecting with a linear layer.

Implement:
  - ConcatBlock(in_a, in_b, out): concatenate x_a (N, in_a) and x_b (N, in_b) along dim=1 → (N, in_a+in_b), then Linear → (N, out).
Run the assert at the bottom to check output shape.
"""

import torch
import torch.nn as nn


class ConcatBlock(nn.Module):
    """Two inputs x_a, x_b → concat along features → linear → output."""

    def __init__(self, in_a: int, in_b: int, out: int):
        super().__init__()
        pass

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    block = ConcatBlock(10, 5, 8)
    x_a, x_b = torch.randn(4, 10), torch.randn(4, 5)
    out = block(x_a, x_b)
    assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"
    print("02 — ConcatBlock: OK.")

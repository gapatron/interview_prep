"""
02 — Conditional MLP (conditioning)
====================================
Learning goal: Feed an extra vector (e.g. class embedding or context) into a classifier
by concatenating it with the hidden representation before the final layer.

Implement:
  - ConditionalMLP(in_dim, c_dim, h, num_classes): first layer Linear(in_dim, h) → ReLU;
    then concatenate hidden (N, h) with condition c (N, c_dim) along dim=1 → (N, h+c_dim);
    then Linear(h+c_dim, num_classes) → logits (N, num_classes).
Run the assert to check output shape.
"""

import torch
import torch.nn as nn


class ConditionalMLP(nn.Module):
    """MLP that takes input x and condition c; hidden state is concatenated with c before the final layer."""

    def __init__(self, in_dim: int, c_dim: int, h: int, num_classes: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    model = ConditionalMLP(in_dim=20, c_dim=4, h=32, num_classes=3)
    x, c = torch.randn(5, 20), torch.randn(5, 4)
    out = model(x, c)
    assert out.shape == (5, 3), f"Expected (5, 3), got {out.shape}"
    print("02 — ConditionalMLP: OK.")

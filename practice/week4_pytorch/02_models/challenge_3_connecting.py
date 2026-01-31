"""
02 — Connecting modules (Encoder → Decoder)
==========================================
Learning goal: Compose two submodules so the output of one is the input of the other—
the standard pattern for encoder–decoder and backbone–head.

Implement:
  - Encoder(in_dim, hidden): Linear(in_dim, hidden) → ReLU. Output (N, hidden).
  - Decoder(hidden, out_dim): Linear(hidden, out_dim). Output (N, out_dim).
  - FullModel(in_dim, hidden, out_dim): forward(x) → decoder(encoder(x)). Store encoder and decoder as submodules.
Run the assert to check output shape.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Maps input to a hidden representation."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Decoder(nn.Module):
    """Maps hidden representation to output."""

    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass


class FullModel(nn.Module):
    """Encoder followed by Decoder; forward(x) = decoder(encoder(x))."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    model = FullModel(in_dim=10, hidden=20, out_dim=2)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"
    print("02 — Encoder/Decoder/FullModel: OK.")

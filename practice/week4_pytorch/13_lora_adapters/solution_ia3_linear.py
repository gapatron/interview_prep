"""
Solution: IA3Linear — base(x) * scale; scale (out_features,); base frozen.
"""

import torch
import torch.nn as nn


class IA3Linear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) * self.scale


if __name__ == "__main__":
    base = nn.Linear(8, 4)
    layer = IA3Linear(base)
    x = torch.randn(3, 8)
    out = layer(x)
    assert out.shape == (3, 4)
    assert not any(p.requires_grad for p in base.parameters())
    out.sum().backward()
    assert layer.scale.grad is not None
    print("IA3Linear OK.")

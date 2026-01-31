"""
Solution: BottleneckAdapter — down_proj -> activation -> up_proj, residual add.
"""

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, residual: bool = True) -> torch.Tensor:
        h = self.down_proj(x)
        h = torch.nn.functional.gelu(h)
        delta = self.up_proj(h)
        return (x + delta) if residual else delta

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Delta only (no residual) for use inside AdapterBlock."""
        return self.forward(x, residual=False)


if __name__ == "__main__":
    B, L, d = 2, 4, 8
    adapter = BottleneckAdapter(d_model=d, bottleneck_dim=4)
    x = torch.randn(B, L, d)
    out = adapter(x)
    assert out.shape == (B, L, d)
    print("BottleneckAdapter OK.")

"""
Solution: LearnablePrefix — prepend learnable (prefix_len, d_model) to (B, L, d) -> (B, prefix_len+L, d).
"""

import torch
import torch.nn as nn


class LearnablePrefix(nn.Module):
    def __init__(self, prefix_len: int, d_model: int):
        super().__init__()
        self.prefix = nn.Parameter(torch.empty(prefix_len, d_model))
        nn.init.normal_(self.prefix, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        prefix_batch = self.prefix.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prefix_batch, x], dim=1)


if __name__ == "__main__":
    B, L, d, prefix_len = 2, 4, 8, 3
    layer = LearnablePrefix(prefix_len=prefix_len, d_model=d)
    x = torch.randn(B, L, d)
    out = layer(x)
    assert out.shape == (B, prefix_len + L, d)
    print("LearnablePrefix OK.")

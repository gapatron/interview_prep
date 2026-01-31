"""
Solution: AdapterBlock — sublayer + adapter delta; forward = x + sublayer(x) + adapter_delta(x + sublayer(x)).
"""

import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    def __init__(self, sublayer: nn.Module, adapter: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.sublayer(x)
        delta = self.adapter.forward_delta(h)
        return h + delta


if __name__ == "__main__":
    from base_model import BaseTransformerBlock
    B, L, d, d_ff = 2, 4, 8, 32
    block = BaseTransformerBlock(d, d_ff)
    from solution_bottleneck_adapter import BottleneckAdapter
    adapter = BottleneckAdapter(d_model=d, bottleneck_dim=4)
    # Wrap the whole block as "sublayer" and add adapter after it
    wrapped = AdapterBlock(block, adapter)
    x = torch.randn(B, L, d)
    out = wrapped(x)
    assert out.shape == (B, L, d)
    print("AdapterBlock OK.")

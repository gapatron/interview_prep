"""
Solution: ParallelAdapterBlock — output = sublayer(x) + adapter_delta(x).
"""

import torch
import torch.nn as nn


class ParallelAdapterBlock(nn.Module):
    def __init__(self, sublayer: nn.Module, adapter: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sublayer(x) + self.adapter.forward_delta(x)


if __name__ == "__main__":
    from base_model import BaseTransformerBlock
    from solution_bottleneck_adapter import BottleneckAdapter
    B, L, d, d_ff = 2, 4, 8, 32
    block = BaseTransformerBlock(d, d_ff)
    adapter = BottleneckAdapter(d_model=d, bottleneck_dim=4)
    wrapped = ParallelAdapterBlock(block, adapter)
    x = torch.randn(B, L, d)
    out = wrapped(x)
    assert out.shape == (B, L, d)
    print("ParallelAdapterBlock OK.")

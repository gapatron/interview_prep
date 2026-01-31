"""
Solution: AdapterStack — multiple BottleneckAdapters; forward(x, adapter_id=k) uses adapter k.
"""

import torch
import torch.nn as nn
from solution_bottleneck_adapter import BottleneckAdapter


class AdapterStack(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int, num_adapters: int):
        super().__init__()
        self.adapters = nn.ModuleList([
            BottleneckAdapter(d_model=d_model, bottleneck_dim=bottleneck_dim)
            for _ in range(num_adapters)
        ])

    def forward(self, x: torch.Tensor, adapter_id: int) -> torch.Tensor:
        return x + self.adapters[adapter_id].forward_delta(x)


if __name__ == "__main__":
    B, L, d, bottleneck, num = 2, 4, 8, 4, 3
    stack = AdapterStack(d_model=d, bottleneck_dim=bottleneck, num_adapters=num)
    x = torch.randn(B, L, d)
    out = stack(x, adapter_id=1)
    assert out.shape == (B, L, d)
    out0 = stack(x, adapter_id=0)
    out2 = stack(x, adapter_id=2)
    assert out0.shape == out2.shape == (B, L, d)
    print("AdapterStack OK.")

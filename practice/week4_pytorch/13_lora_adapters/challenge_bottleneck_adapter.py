"""
Bottleneck Adapter (Houlsby-style)
==================================
Implement a bottleneck adapter: down_proj (d_model -> bottleneck_dim), activation, up_proj (bottleneck_dim -> d_model).
Output = input + up_proj(activation(down_proj(input)))  (residual connection).
Input/output shape: (batch, seq_len, d_model) so the module plugs into transformer blocks.

- down_proj: nn.Linear(d_model, bottleneck_dim)
- up_proj: nn.Linear(bottleneck_dim, d_model)
- Use GELU (or ReLU) between them. Initialize so that initially the adapter output is small (e.g. up_proj zeros)
  so the residual dominates at start.

Implement BottleneckAdapter.__init__ and forward. No solution code — use LoRA_and_Adapters_Guide.md.
"""

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int):
        super().__init__()
        raise NotImplementedError(
            "down_proj: Linear(d_model, bottleneck_dim); up_proj: Linear(bottleneck_dim, d_model). "
            "Init up_proj to zeros so residual dominates at start. forward: x + up(act(down(x)))."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("x + up_proj(activation(down_proj(x))).")

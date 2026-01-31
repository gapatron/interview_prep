"""
Adapter Stack — Multiple adapters (e.g. one per task)
=====================================================
Hold a list of BottleneckAdapters and apply **one** at a time via an index (e.g. task id).
Used for multi-task or multi-tenant: same base, different adapter per task.

- __init__(d_model, bottleneck_dim, num_adapters): create num_adapters BottleneckAdapters (same d_model, bottleneck_dim).
- forward(x, adapter_id): apply adapters[adapter_id] to x (with residual: x + adapter_delta(x)).
  adapter_id is an integer in [0, num_adapters).
- Input/output shape (B, L, d_model).

Implement AdapterStack.__init__ and forward. See LoRA_and_Adapters_Guide.md.
"""

import torch
import torch.nn as nn


class AdapterStack(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int, num_adapters: int):
        super().__init__()
        raise NotImplementedError(
            "Create num_adapters BottleneckAdapters; store in ModuleList. "
            "forward(x, adapter_id): return x + adapters[adapter_id].forward_delta(x)."
        )

    def forward(self, x: torch.Tensor, adapter_id: int) -> torch.Tensor:
        raise NotImplementedError("x + adapters[adapter_id].forward_delta(x).")

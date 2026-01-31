"""
Parallel Adapter Block
======================
Unlike the serial AdapterBlock (sublayer then adapter on sublayer output), a **parallel** adapter
adds the adapter branch on the **input** x, not on the sublayer output:
  output = sublayer(x) + adapter_delta(x)
So the adapter runs in parallel with the sublayer; both see the same input x.

- Use a sublayer (nn.Module, (B, L, d_model) -> (B, L, d_model)) and a BottleneckAdapter.
- adapter_delta(x) = up(activation(down(x))) (no residual inside adapter).
- ParallelAdapterBlock.forward(x) = sublayer(x) + adapter_delta(x).

Implement ParallelAdapterBlock.__init__ and forward. Use your BottleneckAdapter.forward_delta.
See LoRA_and_Adapters_Guide.md — "Parallel adapter".
"""

import torch
import torch.nn as nn


class ParallelAdapterBlock(nn.Module):
    def __init__(self, sublayer: nn.Module, adapter: nn.Module):
        super().__init__()
        raise NotImplementedError(
            "Store sublayer and adapter. Forward: return sublayer(x) + adapter.forward_delta(x)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("sublayer(x) + adapter_delta(x).")

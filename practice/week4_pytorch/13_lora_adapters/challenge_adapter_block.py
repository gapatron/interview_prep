"""
Adapter Block — Insert adapter after a sublayer (e.g. attention or FFN)
========================================================================
Implement a block that wraps a sublayer and a BottleneckAdapter:
  h = x + sublayer(x)
  output = h + adapter_delta(h)
where adapter_delta(h) = up_proj(activation(down_proj(h))) (the adapter’s bottleneck output only, no residual).
So the adapter adds a learned delta on top of the sublayer output; input/output shape (B, L, d_model).

- You need a sublayer (nn.Module that maps (B, L, d_model) -> (B, L, d_model)) and a BottleneckAdapter.
- The adapter’s “delta” is just its bottleneck path: down -> activation -> up (no residual inside this call).
- AdapterBlock.forward(x) = x + sublayer(x) + adapter_delta(x + sublayer(x)).

Implement AdapterBlock.__init__ and forward. Use your BottleneckAdapter; add a method or flag so you can
call “delta only” (up(act(down(x)))) for use here. See LoRA_and_Adapters_Guide.md.
"""

import torch
import torch.nn as nn


class AdapterBlock(nn.Module):
    def __init__(self, sublayer: nn.Module, adapter: nn.Module):
        super().__init__()
        raise NotImplementedError(
            "Store sublayer and adapter. Forward: h = x + sublayer(x); return h + adapter_delta(h). "
            "Adapter must expose delta-only output (e.g. forward_delta or residual=False)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("h = x + sublayer(x); return h + adapter_delta(h).")

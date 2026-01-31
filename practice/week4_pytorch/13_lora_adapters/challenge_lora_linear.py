"""
LoRA — Low-Rank Adaptation Linear Layer
========================================
Implement a Linear layer that wraps an existing nn.Linear and adds a low-rank update:
  output = base_linear(x) + scale * (x @ A.T @ B.T)
where A is (r, in_features), B is (out_features, r), r << min(in_features, out_features).
Only A and B are trained; base_linear is frozen.

- Freeze the original layer (base_linear.requires_grad_(False)).
- LoRA update: delta = B @ A  (shape out_features x in_features), so (x @ A.T @ B.T) = (x @ delta.T).
  Implement as: lora_out = (x @ A.T) @ B.T  or  F.linear(x, B @ A)  so you only store A, B.
- Initialize A (e.g. kaiming_uniform_) and B with zeros so the initial update is zero (training starts from base).
- scaling: often scale = alpha / r for a hyperparameter alpha (default alpha=r so scale=1).

Implement LoRALinear.__init__ and forward. No solution code in docstring — use LoRA_and_Adapters_Guide.md.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        scale: float = 1.0,
    ):
        super().__init__()
        raise NotImplementedError(
            "Store base, freeze it. Create A (r, in_features), B (out_features, r). "
            "Init A (e.g. kaiming), B zeros. forward = base(x) + scale * (x @ A.T @ B.T)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("base(x) + scale * lora_update(x).")

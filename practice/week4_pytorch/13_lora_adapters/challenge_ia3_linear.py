"""
IA³ — Infused Adapter by Inhibiting and Amplifying
==================================================
Wrap an nn.Linear with a **learnable per-output scaling** (elementwise): output = base(x) * scale.
Only the scale vector is trained; base is frozen. Scale shape (out_features,) applied along the last dim.

- base: nn.Linear(in_features, out_features).
- scale: nn.Parameter of shape (out_features,); init to ones so initial behavior = base.
- forward: base(x) * scale (scale broadcasts over batch and any other dims).
- Freeze base.parameters().

Used in T-Few, IA³: very few parameters (one scalar per output dimension). Implement IA3Linear.
See LoRA_and_Adapters_Guide.md — "IA³".
"""

import torch
import torch.nn as nn


class IA3Linear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        raise NotImplementedError(
            "Store base, freeze it. scale: Parameter(out_features,), init ones. forward: base(x) * scale."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("base(x) * scale (broadcast).")

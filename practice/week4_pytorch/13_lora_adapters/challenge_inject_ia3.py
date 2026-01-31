"""
Inject IA³ into a module by name
================================
Replace specified nn.Linear layers with IA3Linear (elementwise scale; base frozen).
Return the modified model and a list of IA³ scale parameters for the optimizer.

- Given: model, linear_names (list of attribute names of Linear layers to wrap).
- For each name, getattr(model, name); if nn.Linear, replace with IA3Linear(layer); setattr(model, name, ia3_layer).
- Return (model, ia3_params) where ia3_params is the list of scale parameters from all injected IA3Linear.

Implement inject_ia3(model, linear_names). See LoRA_and_Adapters_Guide.md — "IA³".
"""

import torch
import torch.nn as nn


def inject_ia3(model: nn.Module, linear_names: list[str]) -> tuple[nn.Module, list[nn.Parameter]]:
    """Replace named nn.Linear with IA3Linear. Return (model, ia3_params)."""
    raise NotImplementedError(
        "For each name in linear_names, getattr(model, name); if nn.Linear, "
        "setattr(model, name, IA3Linear(layer)); collect scale params; return (model, ia3_params)."
    )

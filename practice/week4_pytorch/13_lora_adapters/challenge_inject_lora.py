"""
Inject LoRA into a module
=========================
Implement a function that replaces specified nn.Linear layers in a module with LoRALinear wrappers.
The base linears are frozen; only LoRA parameters (A, B) are trainable.
Return the modified module (in-place or a new wrapper) and a list of LoRA parameters for the optimizer.

- Given: model (nn.Module), linear_names (list of str, the attribute names of Linear layers to wrap),
  rank (int), scale (float).
- For each name in linear_names, get the child: layer = getattr(model, name). If it is nn.Linear,
  replace it with LoRALinear(layer, rank, scale). Replacement can be setattr(model, name, lora_layer).
- Return (model, lora_params) where lora_params is a list of all LoRA parameters (lora_A, lora_B) from
  all injected LoRALinear layers, so the caller can do optimizer = Adam(lora_params).

Implement inject_lora(model, linear_names, rank, scale=1.0). No solution code — use LoRA_and_Adapters_Guide.md.
"""

import torch
import torch.nn as nn


def inject_lora(
    model: nn.Module,
    linear_names: list[str],
    rank: int,
    scale: float = 1.0,
):
    """
    Replace named nn.Linear layers in model with LoRALinear. Return (model, lora_params).
    lora_params: list of Parameters (all lora_A and lora_B) for optimizer.
    """
    raise NotImplementedError(
        "For each name in linear_names, getattr(model, name); if isinstance(..., nn.Linear), "
        "replace with LoRALinear(..., rank, scale) via setattr; collect lora_A, lora_B into lora_params; return (model, lora_params)."
    )

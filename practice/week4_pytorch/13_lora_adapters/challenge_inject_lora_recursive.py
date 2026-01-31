"""
Inject LoRA recursively into a module
=====================================
Replace **all** nn.Linear layers in a module (including nested children) with LoRALinear.
Do not replace by name list; walk the module tree and replace every nn.Linear.
Return the modified model and a list of all LoRA parameters (lora_A, lora_B) for the optimizer.

- Walk modules e.g. via named_modules() or recursive named_children().
- For each nn.Linear (and not already LoRALinear), replace with LoRALinear(linear, rank, scale).
- Replacement: you must set the attribute on the **parent** that points to the child (setattr(parent, child_name, lora_layer)).
- Return (model, lora_params). Skip modules that are already LoRALinear to avoid double wrap.

Implement inject_lora_recursive(model, rank, scale=1.0). See LoRA_and_Adapters_Guide.md.
"""

import torch
import torch.nn as nn


def inject_lora_recursive(
    model: nn.Module,
    rank: int,
    scale: float = 1.0,
):
    """
    Replace all nn.Linear in model (recursively) with LoRALinear. Return (model, lora_params).
    """
    raise NotImplementedError(
        "Walk model (e.g. named_modules); for each nn.Linear get parent and child name; "
        "setattr(parent, child_name, LoRALinear(linear, rank, scale)); collect lora_params; return (model, lora_params)."
    )

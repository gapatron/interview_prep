"""
Solution: inject_lora_recursive — replace all nn.Linear (recursively) with LoRALinear; return (model, lora_params).
"""

import torch
import torch.nn as nn
from solution_lora_linear import LoRALinear


def inject_lora_recursive(
    model: nn.Module,
    rank: int,
    scale: float = 1.0,
) -> tuple[nn.Module, list[nn.Parameter]]:
    lora_params: list[nn.Parameter] = []
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, LoRALinear):
            continue
        parts = full_name.split(".")
        if not parts or parts[-1] == "":
            continue
        parent = model
        for p in parts[:-1]:
            if p:
                parent = getattr(parent, p)
        child_name = parts[-1]
        lora_layer = LoRALinear(module, rank=rank, scale=scale)
        setattr(parent, child_name, lora_layer)
        lora_params.append(lora_layer.lora_A)
        lora_params.append(lora_layer.lora_B)
    return model, lora_params


if __name__ == "__main__":
    from base_model import BaseMLP
    model = BaseMLP(10, 20, 2)
    model, lora_params = inject_lora_recursive(model, rank=4, scale=1.0)
    assert len(lora_params) == 6
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)
    print("inject_lora_recursive OK.")

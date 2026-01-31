"""
Solution: inject_lora — replace named nn.Linear with LoRALinear; return (model, lora_params).
"""

import torch
import torch.nn as nn
from solution_lora_linear import LoRALinear


def inject_lora(
    model: nn.Module,
    linear_names: list[str],
    rank: int,
    scale: float = 1.0,
) -> tuple[nn.Module, list[nn.Parameter]]:
    lora_params: list[nn.Parameter] = []
    for name in linear_names:
        if not hasattr(model, name):
            continue
        layer = getattr(model, name)
        if isinstance(layer, nn.Linear):
            lora_layer = LoRALinear(layer, rank=rank, scale=scale)
            setattr(model, name, lora_layer)
            lora_params.append(lora_layer.lora_A)
            lora_params.append(lora_layer.lora_B)
    return model, lora_params


if __name__ == "__main__":
    from base_model import BaseMLP
    model = BaseMLP(10, 20, 2)
    model, lora_params = inject_lora(model, ["fc1", "fc2"], rank=4, scale=1.0)
    assert len(lora_params) == 4
    base_params = [p for p in model.parameters() if p.requires_grad and p not in lora_params]
    assert all(not p.requires_grad for p in model.fc1.base.parameters())
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)
    print("inject_lora OK.")

"""
Solution: inject_ia3 — replace named nn.Linear with IA3Linear; return (model, ia3_params).
"""

import torch
import torch.nn as nn
from solution_ia3_linear import IA3Linear


def inject_ia3(model: nn.Module, linear_names: list[str]) -> tuple[nn.Module, list[nn.Parameter]]:
    ia3_params: list[nn.Parameter] = []
    for name in linear_names:
        if not hasattr(model, name):
            continue
        layer = getattr(model, name)
        if isinstance(layer, nn.Linear) and not isinstance(layer, IA3Linear):
            ia3_layer = IA3Linear(layer)
            setattr(model, name, ia3_layer)
            ia3_params.append(ia3_layer.scale)
    return model, ia3_params


if __name__ == "__main__":
    from base_model import BaseMLP
    model = BaseMLP(10, 20, 2)
    model, ia3_params = inject_ia3(model, ["fc1", "fc2"])
    assert len(ia3_params) == 2
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)
    print("inject_ia3 OK.")

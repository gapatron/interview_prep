"""
Solution: build_lora_adapter_model — base block + LoRA injection + optional AdapterBlock; return model and trainable_params.
"""

import torch
import torch.nn as nn

from base_model import BaseTransformerBlock
from solution_bottleneck_adapter import BottleneckAdapter
from solution_adapter_block import AdapterBlock
from solution_inject_lora import inject_lora


def build_lora_adapter_model(
    d_model: int,
    d_ff: int,
    lora_rank: int,
    adapter_bottleneck: int,
    use_adapter_block: bool = True,
) -> tuple[nn.Module, list[nn.Parameter]]:
    block = BaseTransformerBlock(d_model, d_ff)
    block, lora_params = inject_lora(
        block,
        linear_names=["proj_in", "proj_out", "ff1", "ff2"],
        rank=lora_rank,
        scale=1.0,
    )
    lora_param_ids = {id(q) for q in lora_params}
    for p in block.parameters():
        if id(p) not in lora_param_ids:
            p.requires_grad_(False)
    trainable_params = list(lora_params)
    if use_adapter_block:
        adapter = BottleneckAdapter(d_model=d_model, bottleneck_dim=adapter_bottleneck)
        block = AdapterBlock(block, adapter)
        trainable_params += list(adapter.parameters())
    return block, trainable_params


if __name__ == "__main__":
    d_model, d_ff, rank, bottleneck = 8, 32, 2, 4
    model, trainable = build_lora_adapter_model(d_model, d_ff, rank, bottleneck, use_adapter_block=True)
    B, L = 2, 4
    x = torch.randn(B, L, d_model)
    out = model(x)
    assert out.shape == (B, L, d_model)
    base_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train_count = sum(p.numel() for p in trainable)
    assert train_count < base_count
    out.sum().backward()
    for p in trainable:
        assert p.grad is not None
    print("build_lora_adapter_model OK.")

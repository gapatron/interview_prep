"""
Full model: Base + LoRA + Adapters
==================================
Build a model that connects LoRA and adapters to a base transformer block:
1. Create a base block (BaseTransformerBlock from base_model.py).
2. Inject LoRA into specified Linear layers (e.g. proj_in, proj_out, ff1, ff2) with a given rank.
3. Optionally wrap the block in an AdapterBlock (block as sublayer, BottleneckAdapter as adapter).
4. Freeze all base parameters; only LoRA (A, B) and adapter parameters are trainable.
5. Forward must work: input (B, L, d_model) -> output (B, L, d_model). Backward must flow to LoRA and adapter params.

Implement build_lora_adapter_model(d_model, d_ff, lora_rank, adapter_bottleneck, use_adapter_block=True)
returning (model, trainable_params). trainable_params is a list of all parameters that should be trained
(LoRA + adapter). Base block linears (before LoRA wrap) must be frozen.

No solution code — use LoRA_and_Adapters_Guide.md and the solutions for LoRALinear, inject_lora, BottleneckAdapter, AdapterBlock.
"""

import torch
import torch.nn as nn


def build_lora_adapter_model(
    d_model: int,
    d_ff: int,
    lora_rank: int,
    adapter_bottleneck: int,
    use_adapter_block: bool = True,
):
    """
    Return (model, trainable_params).
    model: forward(x) -> (B, L, d_model).
    trainable_params: list of Parameters (LoRA + adapter only).
    """
    raise NotImplementedError(
        "Create BaseTransformerBlock; inject_lora into proj_in, proj_out, ff1, ff2; "
        "optionally wrap in AdapterBlock(block, BottleneckAdapter); freeze base; collect trainable_params."
    )

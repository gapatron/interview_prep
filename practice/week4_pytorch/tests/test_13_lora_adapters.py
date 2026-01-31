"""Tests for 13_lora_adapters solutions: LoRA, BottleneckAdapter, AdapterBlock, inject_lora, build_lora_adapter_model."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "13_lora_adapters"))


def test_lora_linear_shape():
    from solution_lora_linear import LoRALinear
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, rank=2, scale=1.0)
    x = torch.randn(3, 8)
    out = lora(x)
    assert out.shape == (3, 4)


def test_lora_linear_base_frozen():
    from solution_lora_linear import LoRALinear
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, rank=2, scale=1.0)
    assert not any(p.requires_grad for p in base.parameters())
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_lora_linear_backward():
    from solution_lora_linear import LoRALinear
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, rank=2, scale=1.0)
    x = torch.randn(3, 8, requires_grad=True)
    out = lora(x)
    out.sum().backward()
    assert x.grad is not None
    assert lora.lora_A.grad is not None and lora.lora_B.grad is not None


def test_bottleneck_adapter_shape():
    from solution_bottleneck_adapter import BottleneckAdapter
    B, L, d = 2, 4, 8
    adapter = BottleneckAdapter(d_model=d, bottleneck_dim=4)
    x = torch.randn(B, L, d)
    out = adapter(x)
    assert out.shape == (B, L, d)


def test_bottleneck_adapter_forward_delta():
    from solution_bottleneck_adapter import BottleneckAdapter
    B, L, d = 2, 4, 8
    adapter = BottleneckAdapter(d_model=d, bottleneck_dim=4)
    x = torch.randn(B, L, d)
    delta = adapter.forward_delta(x)
    assert delta.shape == (B, L, d)


def test_adapter_block_shape():
    from base_model import BaseTransformerBlock
    from solution_bottleneck_adapter import BottleneckAdapter
    from solution_adapter_block import AdapterBlock
    B, L, d, d_ff = 2, 4, 8, 32
    block = BaseTransformerBlock(d, d_ff)
    adapter = BottleneckAdapter(d_model=d, bottleneck_dim=4)
    wrapped = AdapterBlock(block, adapter)
    x = torch.randn(B, L, d)
    out = wrapped(x)
    assert out.shape == (B, L, d)


def test_inject_lora_returns_model_and_params():
    from base_model import BaseMLP
    from solution_inject_lora import inject_lora
    model = BaseMLP(10, 20, 2)
    model, lora_params = inject_lora(model, ["fc1", "fc2"], rank=4, scale=1.0)
    assert len(lora_params) == 4
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)


def test_inject_lora_base_frozen():
    from base_model import BaseMLP
    from solution_inject_lora import inject_lora
    model = BaseMLP(10, 20, 2)
    model, lora_params = inject_lora(model, ["fc1", "fc2"], rank=4, scale=1.0)
    assert not any(p.requires_grad for p in model.fc1.base.parameters())
    assert not any(p.requires_grad for p in model.fc2.base.parameters())


def test_build_lora_adapter_model_shape():
    from solution_lora_adapter_model import build_lora_adapter_model
    d_model, d_ff, rank, bottleneck = 8, 32, 2, 4
    model, trainable = build_lora_adapter_model(d_model, d_ff, rank, bottleneck, use_adapter_block=True)
    B, L = 2, 4
    x = torch.randn(B, L, d_model)
    out = model(x)
    assert out.shape == (B, L, d_model)


def test_build_lora_adapter_model_trainable_less_than_total():
    from solution_lora_adapter_model import build_lora_adapter_model
    d_model, d_ff, rank, bottleneck = 8, 32, 2, 4
    model, trainable = build_lora_adapter_model(d_model, d_ff, rank, bottleneck, use_adapter_block=True)
    total = sum(p.numel() for p in model.parameters())
    train_count = sum(p.numel() for p in trainable)
    assert train_count < total


def test_build_lora_adapter_model_backward():
    from solution_lora_adapter_model import build_lora_adapter_model
    d_model, d_ff, rank, bottleneck = 8, 32, 2, 4
    model, trainable = build_lora_adapter_model(d_model, d_ff, rank, bottleneck, use_adapter_block=True)
    B, L = 2, 4
    x = torch.randn(B, L, d_model)
    out = model(x)
    out.sum().backward()
    for p in trainable:
        assert p.grad is not None


def test_base_mlp_shape():
    from base_model import BaseMLP
    model = BaseMLP(10, 20, 2)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 2)


def test_base_transformer_block_shape():
    from base_model import BaseTransformerBlock
    B, L, d, d_ff = 2, 4, 8, 32
    block = BaseTransformerBlock(d, d_ff)
    x = torch.randn(B, L, d)
    out = block(x)
    assert out.shape == (B, L, d)


# ---- Parallel adapter ----
def test_parallel_adapter_block_shape():
    from base_model import BaseTransformerBlock
    from solution_bottleneck_adapter import BottleneckAdapter
    from solution_parallel_adapter_block import ParallelAdapterBlock
    B, L, d, d_ff = 2, 4, 8, 32
    block = BaseTransformerBlock(d, d_ff)
    adapter = BottleneckAdapter(d_model=d, bottleneck_dim=4)
    wrapped = ParallelAdapterBlock(block, adapter)
    x = torch.randn(B, L, d)
    out = wrapped(x)
    assert out.shape == (B, L, d)


# ---- Learnable prefix ----
def test_learnable_prefix_shape():
    from solution_learnable_prefix import LearnablePrefix
    B, L, d, prefix_len = 2, 4, 8, 3
    layer = LearnablePrefix(prefix_len=prefix_len, d_model=d)
    x = torch.randn(B, L, d)
    out = layer(x)
    assert out.shape == (B, prefix_len + L, d)


# ---- IA³ ----
def test_ia3_linear_shape():
    from solution_ia3_linear import IA3Linear
    base = nn.Linear(8, 4)
    layer = IA3Linear(base)
    x = torch.randn(3, 8)
    out = layer(x)
    assert out.shape == (3, 4)


def test_ia3_linear_base_frozen():
    from solution_ia3_linear import IA3Linear
    base = nn.Linear(8, 4)
    layer = IA3Linear(base)
    assert not any(p.requires_grad for p in base.parameters())
    assert layer.scale.requires_grad


def test_inject_ia3_returns_model_and_params():
    from base_model import BaseMLP
    from solution_inject_ia3 import inject_ia3
    model = BaseMLP(10, 20, 2)
    model, ia3_params = inject_ia3(model, ["fc1", "fc2"])
    assert len(ia3_params) == 2
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)


# ---- Recursive LoRA injection ----
def test_inject_lora_recursive_returns_model_and_params():
    from base_model import BaseMLP
    from solution_inject_lora_recursive import inject_lora_recursive
    model = BaseMLP(10, 20, 2)
    model, lora_params = inject_lora_recursive(model, rank=4, scale=1.0)
    assert len(lora_params) == 6
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)


# ---- Adapter stack ----
def test_adapter_stack_shape():
    from solution_adapter_stack import AdapterStack
    B, L, d, bottleneck, num = 2, 4, 8, 4, 3
    stack = AdapterStack(d_model=d, bottleneck_dim=bottleneck, num_adapters=num)
    x = torch.randn(B, L, d)
    out = stack(x, adapter_id=1)
    assert out.shape == (B, L, d)


def test_adapter_stack_different_ids():
    from solution_adapter_stack import AdapterStack
    B, L, d, bottleneck, num = 2, 4, 8, 4, 3
    stack = AdapterStack(d_model=d, bottleneck_dim=bottleneck, num_adapters=num)
    x = torch.randn(B, L, d)
    out0 = stack(x, adapter_id=0)
    out2 = stack(x, adapter_id=2)
    assert out0.shape == out2.shape == (B, L, d)

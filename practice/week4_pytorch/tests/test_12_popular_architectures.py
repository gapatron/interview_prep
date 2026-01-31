"""Tests for 12_popular_architectures solutions."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "12_popular_architectures"))


def test_residual_block_shape_stride1():
    from solution_residual_block import ResidualBlock
    block = ResidualBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)


def test_residual_block_shape_stride2():
    from solution_residual_block import ResidualBlock
    block = ResidualBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == (2, 128, 4, 4)


def test_simple_cnn_shape():
    from solution_simple_cnn import SimpleCNN
    model = SimpleCNN(3, 10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    assert not torch.isnan(out).any()


def test_vgg_block_shape():
    from solution_vgg_block import VGGBlock
    block = VGGBlock(3, 64, num_convs=2, use_pool=True)
    x = torch.randn(2, 3, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)


def test_make_layer_shape():
    from solution_make_layer import make_layer
    layer = make_layer(64, 128, num_blocks=2, stride=2)
    x = torch.randn(2, 64, 8, 8)
    out = layer(x)
    assert out.shape == (2, 128, 4, 4)


def test_bottleneck_shape():
    from solution_bottleneck import BottleneckBlock
    block = BottleneckBlock(256, 512, 128, stride=2)
    x = torch.randn(2, 256, 8, 8)
    out = block(x)
    assert out.shape == (2, 512, 4, 4)


def test_small_resnet_shape():
    from solution_small_resnet import SmallResNet
    model = SmallResNet(10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_inception_module_shape():
    from solution_inception_module import InceptionModule
    mod = InceptionModule(64, 16, 16, 32, 16, 16, 16)
    x = torch.randn(2, 64, 8, 8)
    out = mod(x)
    assert out.shape == (2, 80, 8, 8), f"Expected (2, 80, 8, 8), got {out.shape}"

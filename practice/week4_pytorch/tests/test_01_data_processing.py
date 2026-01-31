"""Tests for 01_data_processing solutions. Validates shapes, dtypes, and correctness."""

import sys
from pathlib import Path

import pytest
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "01_data_processing"))


@pytest.fixture(autouse=True)
def _path():
    sys.path.insert(0, str(BASE / "01_data_processing"))


def test_get_device():
    from solution_1_tensors import get_device
    device = get_device()
    assert device is not None
    assert device.type in ("cuda", "cpu")


def test_create_batch_tensor():
    from solution_1_tensors import create_batch_tensor
    t = create_batch_tensor(4, 3, 8, 8)
    assert t.shape == (4, 3, 8, 8), f"Expected (4,3,8,8), got {t.shape}"
    assert t.dtype == torch.float32
    assert not torch.isnan(t).any() and not torch.isinf(t).any()


def test_concat_along_channels():
    from solution_1_tensors import concat_along_channels
    a = torch.randn(2, 3, 4, 4)
    b = torch.randn(2, 5, 4, 4)
    c = concat_along_channels(a, b)
    assert c.shape == (2, 8, 4, 4), f"Expected (2,8,4,4), got {c.shape}"
    assert c.dim() == 4


def test_condition_to_feature_map():
    from solution_1_tensors import condition_to_feature_map
    x = torch.randn(2, 3, 4, 4)
    cond = torch.randn(2, 6)
    out = condition_to_feature_map(x, cond)
    assert out.shape == (2, 9, 4, 4), f"Expected (2,9,4,4), got {out.shape}"
    assert out.size(1) == x.size(1) + cond.size(1)


def test_stack_batches():
    from solution_1_tensors import stack_batches
    tensors = [torch.randn(3, 4, 4) for _ in range(5)]
    stacked = stack_batches(tensors)
    assert stacked.shape == (5, 3, 4, 4), f"Expected (5,3,4,4), got {stacked.shape}"
    assert stacked.dim() == 4


def test_tensor_dataset_len_and_getitem():
    from solution_2_dataset import TensorDataset
    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10,))
    ds = TensorDataset(X, y)
    assert len(ds) == 10
    x, y0 = ds[0]
    assert x.shape == (5,)
    assert y0.ndim == 0
    assert y0.dtype in (torch.int64, torch.long)


def test_create_sample_loader():
    from solution_2_dataset import create_sample_loader
    loader = create_sample_loader(batch_size=4, num_samples=20)
    batches = list(loader)
    assert len(batches) >= 1
    bx, by = batches[0]
    assert bx.dim() == 2 and bx.size(0) <= 4
    assert by.dtype in (torch.int64, torch.long), "Labels must be long for CrossEntropyLoss"
    assert by.shape[0] == bx.shape[0]

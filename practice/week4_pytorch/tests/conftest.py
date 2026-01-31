"""
Pytest configuration and shared fixtures for week4_pytorch solution tests.
Run from practice/week4_pytorch: pytest tests/ -v
Or from repo root: pytest practice/week4_pytorch/tests/ -v
"""

import sys
from pathlib import Path

import pytest
import torch

# week4_pytorch root
BASE = Path(__file__).resolve().parent.parent


def _add_path(subdir: str) -> None:
    """Prepend a solution subdir to sys.path so we can import solution_* modules."""
    d = BASE / subdir
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))


@pytest.fixture
def device():
    """Shared device: cuda if available else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set and return a fixed seed for reproducibility in tests."""
    s = 42
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


@pytest.fixture
def sample_4d():
    """Sample 4D tensors (N, C, H, W) for concat/condition tests."""
    return torch.randn(2, 3, 4, 4), torch.randn(2, 5, 4, 4)


@pytest.fixture
def sample_2d():
    """Sample 2D tensors (N, D) for linear/MLP tests."""
    return torch.randn(4, 10)

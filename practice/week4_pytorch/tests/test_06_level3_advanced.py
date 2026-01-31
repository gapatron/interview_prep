"""Tests for 06_level3_advanced solutions: multi-input, gradient clip, accumulation, mixed precision, checkpointing, dataset."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "06_level3_advanced"))


def test_multi_input_classifier_shape():
    from solution_multi_input import MultiInputClassifier
    model = MultiInputClassifier(3, 8, 8, 5, 64, 32, 4)
    img = torch.randn(2, 3, 8, 8)
    meta = torch.randn(2, 5)
    out = model(img, meta)
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"


def test_gradient_clip_runs_and_returns_float():
    from solution_gradient_clip import train_one_epoch_with_clip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))), batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch_with_clip(model, loader, criterion, optimizer, device, max_norm=1.0)
    assert isinstance(loss, float) and loss >= 0 and not (loss != loss)


def test_gradient_accumulation_returns_finite_loss():
    from solution_gradient_accumulation import train_one_epoch_accum
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))), batch_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch_accum(model, loader, criterion, optimizer, device, accum_steps=2)
    assert isinstance(loss, float) and not (loss != loss) and loss >= 0


def test_mixed_precision_runs():
    from solution_mixed_precision import train_one_epoch_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))), batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch_amp(model, loader, criterion, optimizer, device)
    assert isinstance(loss, float) and loss >= 0 and not (loss != loss)


def test_gradient_checkpointing_backward():
    from solution_gradient_checkpointing import CheckpointedBlock
    model = CheckpointedBlock(8, 16)
    x = torch.randn(2, 8, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    assert out.shape == (2, 8)


def test_path_label_dataset():
    from solution_custom_dataset_paths import PathLabelDataset
    path_to_tensor = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    pairs = [("a", 0), ("b", 1)]
    ds = PathLabelDataset(pairs, path_to_tensor)
    assert len(ds) == 2
    x, y = ds[1]
    assert x.tolist() == [3.0, 4.0] and y == 1

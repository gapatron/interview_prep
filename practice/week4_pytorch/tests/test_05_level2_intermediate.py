"""Tests for 05_level2_intermediate solutions."""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "05_level2_intermediate"))


def test_validate_returns_reasonable_acc():
    from solution_train_val_loop import validate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    loss, acc = validate(model, loader, criterion, device)
    assert isinstance(loss, float) and isinstance(acc, float)
    assert 0 <= acc <= 1, f"Accuracy must be in [0,1], got {acc}"
    assert loss >= 0 and not (loss != loss), "Loss must be non-negative and not NaN"


def test_train_for_epochs_runs():
    from solution_train_val_loop import train_for_epochs
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    train_loader = DataLoader(TensorDataset(torch.randn(64, 10), torch.randint(0, 2, (64,))), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))), batch_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_for_epochs(model, train_loader, val_loader, criterion, optimizer, device, 2)


def test_backbone_head_classifier_shape():
    from solution_backbone_head import BackboneHeadClassifier
    model = BackboneHeadClassifier(c_in=3, h=8, w=8, hidden=64, num_classes=5)
    x = torch.randn(4, 3, 8, 8)
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
    assert not torch.isnan(out).any()


def test_checkpoint_save_load_roundtrip():
    from solution_checkpoint import save_checkpoint, load_checkpoint
    import tempfile
    model = nn.Linear(2, 2)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(model, opt, 3, path)
        model2 = nn.Linear(2, 2)
        opt2 = optim.Adam(model2.parameters(), lr=1e-3)
        epoch = load_checkpoint(path, model2, opt2)
        assert epoch == 3
        # State should be loaded
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert "model" in state and "epoch" in state
    finally:
        Path(path).unlink(missing_ok=True)

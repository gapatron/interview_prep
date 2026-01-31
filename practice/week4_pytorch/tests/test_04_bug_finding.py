"""Tests for 04_bug_finding solutions. Validates that fixes run and produce valid output."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "04_bug_finding"))


def test_solution_bug_1_runs():
    from solution_bug_1 import main
    main()


def test_solution_bug_2_concat_shape():
    from solution_bug_2 import ConcatBlock
    block = ConcatBlock(10, 5, 8)
    x_a = torch.randn(4, 10)
    x_b = torch.randn(4, 5)
    out = block(x_a, x_b)
    assert out.shape == (4, 8), f"Bug 2 fix: expected (4,8), got {out.shape}"


def test_solution_bug_3_condition_shape():
    from solution_bug_3 import condition_to_feature_map
    x = torch.randn(2, 3, 4, 4)
    cond = torch.randn(2, 6)
    out = condition_to_feature_map(x, cond)
    assert out.shape == (2, 9, 4, 4)


def test_solution_bug_4_runs():
    from solution_bug_4 import main
    main()


def test_solution_bug_5_validate_returns_tuple():
    from solution_bug_5 import validate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))), batch_size=8)
    loss, acc = validate(model, loader, nn.CrossEntropyLoss(), device)
    assert isinstance(loss, float) and isinstance(acc, float)
    assert 0 <= acc <= 1, f"Accuracy must be in [0,1], got {acc}"


def test_solution_bug_6_backbone_head_shape():
    from solution_bug_6 import BuggyModel  # actually the fixed model
    model = BuggyModel()
    x = torch.randn(4, 3, 8, 8)
    out = model(x)
    assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"


def test_solution_bug_7_backward_succeeds():
    from solution_bug_7 import BuggyBlock
    model = BuggyBlock(4)
    x = torch.randn(2, 4, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


def test_solution_bug_8_targets_shape_and_loss_scalar():
    # Bug 8 fix: targets (N,) long; loss must be scalar
    model = nn.Linear(10, 3)
    logits = model(torch.randn(4, 10))
    targets = torch.randint(0, 3, (4,), dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    assert loss.dim() == 0, "Loss must be scalar for backward"
    assert not torch.isnan(loss).any()
    assert targets.dim() == 1 and targets.dtype in (torch.int64, torch.long)


def test_solution_bug_9_validate_eval():
    from solution_bug_9 import validate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loader = DataLoader(TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,))), batch_size=8)
    acc = validate(model, loader, device)
    assert 0 <= acc <= 1


def test_solution_bug_10_scheduler_step_once_per_epoch():
    # Solution steps scheduler once per epoch; after 3 steps LR = 0.0125
    import torch.optim as optim
    model = nn.Linear(4, 2)
    opt = optim.SGD(model.parameters(), lr=0.1)
    sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    for _ in range(3):
        sched.step()
    lr = opt.param_groups[0]["lr"]
    assert abs(lr - 0.0125) < 1e-6, f"After 3 steps: expected 0.0125, got {lr}"

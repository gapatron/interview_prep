"""
05 — Save and load checkpoint
=============================
Learning goal: Persist model (and optionally optimizer) state so you can resume training or deploy the best model. Industry pattern: save the checkpoint when validation improves.

Implement:
  - save_checkpoint(model, optimizer, epoch, path): save a dict with model.state_dict(), optimizer.state_dict(), and epoch; torch.save to path.
  - load_checkpoint(path, model, optimizer=None): torch.load with map_location="cpu"; model.load_state_dict(...); if optimizer given, load its state_dict; return epoch.

Run the block at the bottom to verify save/load round-trip.
"""

import torch
import torch.nn as nn
from pathlib import Path


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
):
    """Save model state_dict, optimizer state_dict, and epoch."""
    # TODO: state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    # TODO: torch.save(state, path)
    pass


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load checkpoint; optionally load optimizer. Return epoch."""
    # TODO: state = torch.load(path, map_location="cpu")
    # TODO: model.load_state_dict(state["model"])
    # TODO: if optimizer is not None and "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
    # TODO: return state.get("epoch", 0)
    pass


if __name__ == "__main__":
    model = nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = Path("/tmp/pytorch_checkpoint_test.pt")
    save_checkpoint(model, opt, 3, path)
    model2 = nn.Linear(2, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    epoch = load_checkpoint(path, model2, opt2)
    assert epoch == 3
    path.unlink(missing_ok=True)
    print("Checkpoint save/load OK.")

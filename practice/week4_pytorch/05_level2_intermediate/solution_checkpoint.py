"""Solution: Save / load checkpoint."""

import torch
import torch.nn as nn
from pathlib import Path


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str | Path):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", 0)

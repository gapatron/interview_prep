"""
06 — Gradient checkpointing (trade compute for memory)
======================================================
Learning goal: Gradient checkpointing recomputes activations in backward instead of storing
them, so memory use goes down and compute goes up. Use for large layers or long sequences.

Implement:
  - CheckpointedBlock: a module with two submodules (e.g. two Linear layers). In forward,
    run the first submodule normally; run the second submodule inside torch.utils.checkpoint.checkpoint
    so it is recomputed in backward. Signature: checkpoint(fn, *args) or checkpoint(fn, input).
  - So: h = self.layer1(x); out = checkpoint(self._layer2_fn, h, use_reentrant=False) (use_reentrant=False is recommended in PyTorch 2.x).
  - The callable passed to checkpoint must take the saved tensors as args and return the layer output.

Run the assert at the bottom. See PyTorch_Mastery_Guide.md “Gradient checkpointing” and “Quizzes.”
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedBlock(nn.Module):
    """Two layers; the second is run inside checkpoint to save memory."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden)
        self.layer2 = nn.Linear(hidden, dim)

    def _layer2_fn(self, h: torch.Tensor) -> torch.Tensor:
        return self.layer2(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: h = self.layer1(x); h = torch.relu(h)
        # TODO: out = checkpoint(self._layer2_fn, h)  # recomputes layer2 in backward
        # TODO: return out
        pass


if __name__ == "__main__":
    model = CheckpointedBlock(8, 16)
    x = torch.randn(2, 8, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None and out.shape == (2, 8)
    print("Gradient checkpointing OK.")

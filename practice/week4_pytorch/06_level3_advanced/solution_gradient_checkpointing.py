"""
Solution: Gradient checkpointing — wrap a submodule in checkpoint() to recompute in backward.
Takeaway: checkpoint(fn, *args) runs fn(*args) in forward but recomputes it in backward; saves memory, costs compute.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedBlock(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden)
        self.layer2 = nn.Linear(hidden, dim)

    def _layer2_fn(self, h: torch.Tensor) -> torch.Tensor:
        return self.layer2(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.layer1(x))
        out = checkpoint(self._layer2_fn, h, use_reentrant=False)
        return out


if __name__ == "__main__":
    model = CheckpointedBlock(8, 16)
    x = torch.randn(2, 8, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None and out.shape == (2, 8)
    print("Gradient checkpointing OK.")

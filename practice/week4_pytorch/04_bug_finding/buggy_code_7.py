"""
Bug 7 — In-place op breaking autograd
--------------------------------------
Concept: In-place ops (e.g. x.relu_()) can modify tensors that are needed for backward; PyTorch may raise or give wrong gradients.
What goes wrong: An in-place activation is used on a tensor that requires grad; use out-of-place (e.g. torch.relu(x)) instead. Run, find, fix. Compare with solution_bug_7.py.
"""

import torch
import torch.nn as nn


class BuggyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        # BUG: in-place relu_ modifies x which may need grad for backward
        x = self.linear(x)
        x.relu_()
        return x


if __name__ == "__main__":
    model = BuggyBlock(4)
    x = torch.randn(2, 4, requires_grad=True)
    out = model(x)
    out.sum().backward()
    print("OK.")

"""
Solution: Bug 7 — In-place op.
Fix: Use out-of-place torch.relu(x) instead of x.relu_() on tensors that need grad.
Why: In-place ops can invalidate the graph needed for backward; out-of-place preserves it.
"""

import torch
import torch.nn as nn


class BuggyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x


if __name__ == "__main__":
    model = BuggyBlock(4)
    x = torch.randn(2, 4, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    print("OK.")

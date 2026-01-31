"""
Bug 6 — Shape mismatch in forward (backbone → head)
---------------------------------------------------
Concept: Backbone maps (N,C,H,W) → (N, hidden); head maps (N, hidden) → logits. The full model must pass backbone output to head, not raw input.
What goes wrong: forward passes x directly to head instead of backbone(x); head expects (N, hidden) and gets (N,C,H,W). Run, find, fix. Compare with solution_bug_6.py.
"""

import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, flat_size, hidden):
        super().__init__()
        self.fc = nn.Linear(flat_size, hidden)

    def forward(self, x):
        N = x.size(0)
        return torch.relu(self.fc(x.view(N, -1)))


class Head(nn.Module):
    def __init__(self, hidden, num_classes):
        super().__init__()
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, z):
        return self.fc(z)


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(3 * 8 * 8, 64)
        self.head = Head(64, 2)

    def forward(self, x):
        # BUG: we pass x (N, 3, 8, 8) to head which expects (N, 64)
        return self.head(x)


if __name__ == "__main__":
    model = BuggyModel()
    x = torch.randn(4, 3, 8, 8)
    out = model(x)
    print(out.shape)

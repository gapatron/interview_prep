"""
Solution: Bug 6 — Backbone → head shape.
Fix: forward(x) = self.head(self.backbone(x)); do not pass x directly to head.
Why: Head expects (N, hidden); backbone outputs that from (N,C,H,W). Raw x has the wrong shape.
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
        z = self.backbone(x)
        return self.head(z)


if __name__ == "__main__":
    model = BuggyModel()
    x = torch.randn(4, 3, 8, 8)
    out = model(x)
    assert out.shape == (4, 2), out.shape
    print("OK.")

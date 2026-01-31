"""Solution: Backbone + Head classifier."""

import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, c_in: int, h: int, w: int, hidden: int):
        super().__init__()
        self.flatten_size = c_in * h * w
        self.fc = nn.Linear(self.flatten_size, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x = x.view(N, -1)
        return torch.relu(self.fc(x))


class Head(nn.Module):
    def __init__(self, hidden: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class BackboneHeadClassifier(nn.Module):
    def __init__(self, c_in: int, h: int, w: int, hidden: int, num_classes: int):
        super().__init__()
        self.backbone = Backbone(c_in, h, w, hidden)
        self.head = Head(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

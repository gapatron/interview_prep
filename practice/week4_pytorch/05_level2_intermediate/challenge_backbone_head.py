"""
05 — Backbone + head classifier
================================
Learning goal: Structure a classifier as feature extractor (backbone) plus classifier (head). Standard pattern for transfer learning and modular design.

Implement:
  - Backbone: (N, C_in, H, W) → flatten → Linear(C_in*H*W, hidden) → ReLU → (N, hidden).
  - Head: (N, hidden) → Linear(hidden, num_classes) → (N, num_classes).
  - BackboneHeadClassifier: store backbone and head; forward(x) = head(backbone(x)).

Run the assert to check output shape.
"""

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """Flatten spatial dims and project to hidden size."""

    def __init__(self, c_in: int, h: int, w: int, hidden: int):
        super().__init__()
        self.flatten_size = c_in * h * w
        # TODO: self.fc = nn.Linear(self.flatten_size, hidden)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> flatten to (N, C*H*W)
        # TODO: N, C, H, W = x.shape; x = x.view(N, -1); return torch.relu(self.fc(x))
        pass


class Head(nn.Module):
    """hidden -> num_classes."""

    def __init__(self, hidden: int, num_classes: int):
        super().__init__()
        # TODO: self.fc = nn.Linear(hidden, num_classes)
        pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: return self.fc(z)
        pass


class BackboneHeadClassifier(nn.Module):
    """Plug backbone output into head."""

    def __init__(self, c_in: int, h: int, w: int, hidden: int, num_classes: int):
        super().__init__()
        # TODO: self.backbone = Backbone(c_in, h, w, hidden)
        # TODO: self.head = Head(hidden, num_classes)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return self.head(self.backbone(x))
        pass


if __name__ == "__main__":
    model = BackboneHeadClassifier(c_in=3, h=8, w=8, hidden=64, num_classes=5)
    x = torch.randn(4, 3, 8, 8)
    out = model(x)
    assert out.shape == (4, 5), out.shape
    print("BackboneHeadClassifier OK.")

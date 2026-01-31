"""
Popular Architectures — Small ResNet (Full Forward)
===================================================
Implement a small ResNet that maps (N, 3, 32, 32) to (N, num_classes) logits.

Structure:
- Stem: Conv2d(3, 64, 7, stride=2, padding=3) → BN → ReLU → MaxPool2d(2). So 32→16→8.
- Layer1: 2 × ResidualBlock(64, 64, stride=1). Still 8×8, 64 ch.
- Layer2: first ResidualBlock(64, 128, stride=2), then ResidualBlock(128, 128, stride=1). So 8→4, 128 ch.
- Layer3: first ResidualBlock(128, 256, stride=2), then ResidualBlock(256, 256, stride=1). So 4→2, 256 ch.
- Global pool: AdaptiveAvgPool2d(1) → flatten → Linear(256, num_classes).

Use make_layer from challenge_make_layer (or equivalent) or build the layers explicitly.
Ensure every residual add has matching shapes (shortcuts when stride or channels change).

Read: Popular_Architectures_Explained.md — “ResNet Layer”, “Common Bugs”, “Checklist”.

Implement SmallResNet. No answer code — derive from the guide and shape asserts.
"""

import torch
import torch.nn as nn


class SmallResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        raise NotImplementedError("Stem; layer1 (64,2 blocks); layer2 (64->128,2 blocks); layer3 (128->256,2 blocks); pool; fc(256,num_classes).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("stem -> layer1 -> layer2 -> layer3 -> pool -> flatten -> fc. Return logits.")


if __name__ == "__main__":
    model = SmallResNet(10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    print("SmallResNet OK.")

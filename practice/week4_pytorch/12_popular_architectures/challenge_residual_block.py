"""
Popular Architectures — Residual Block (ResNet-style)
=====================================================
Implement a residual block: main path (two 3×3 convs with BN and ReLU) plus a shortcut;
output = ReLU(main_path_output + shortcut(x)).

- When stride == 1 and in_c == out_c, the shortcut can be identity (x unchanged).
- When stride != 1 or in_c != out_c, the main path changes spatial size and/or channels,
  so you must project x to the same shape before adding — otherwise you get a shape mismatch.
  Use a 1×1 conv (with the same stride as the first conv) and BN for the shortcut.

Read: practice/study_guides/Popular_Architectures_Explained.md
  — "Why Residual Connections", "Why Shortcut When stride≠1 or in_c≠out_c", "Common Bugs".

Implement ResidualBlock.forward and, if needed, the shortcut in __init__.
Do not rely on TODOs that give the code — use the Explained guide for the reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        raise NotImplementedError("Define shortcut: identity when shape already matches, else 1×1 conv+BN.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Main path: conv1->bn1->relu, conv2->bn2. Then add shortcut(x), then ReLU.")


if __name__ == "__main__":
    block = ResidualBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)
    block2 = ResidualBlock(64, 128, stride=2)
    out2 = block2(x)
    assert out2.shape == (2, 128, 4, 4)
    print("ResidualBlock OK.")

"""
Popular Architectures — Simple CNN from Scratch
=================================================
Implement a small CNN that maps (N, 3, 32, 32) to (N, num_classes) logits.

Structure: a stem (conv 3→32, BN, ReLU, MaxPool 2×2), then two more blocks.
Each block: conv (with in_ch → out_ch), BN, ReLU, MaxPool 2×2.
Then global average pool (so you get one value per channel), flatten, then a linear layer to num_classes.

- Use 3×3 convs with padding=1 so spatial size is unchanged until you pool.
- MaxPool2d(2) halves H and W. So 32→16→8→4 if you have stem + 2 blocks with pool each.
- The linear layer’s in_features must equal the number of channels after the last conv
  (e.g. if the last block has 128 channels, linear is Linear(128, num_classes)).

Read: practice/study_guides/Popular_Architectures_Explained.md
  — "Why Stacks of 3×3", "VGG-Style", "Common Bugs" (channel progression, linear input size).

Implement SimpleCNN.__init__ and forward. No answer code here — use the guide for the why.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        raise NotImplementedError("Build stem and two blocks (conv->BN->ReLU->Pool); then global pool and fc.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Run stem, block1, block2, pool, flatten, fc. Return logits (no softmax).")


if __name__ == "__main__":
    model = SimpleCNN(3, 10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    print("SimpleCNN OK.")

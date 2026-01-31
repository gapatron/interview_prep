"""
Popular Architectures — VGG-Style Conv Block
=============================================
Implement a single “VGG block”: a sequence of (Conv 3×3, padding=1 → BN → ReLU) repeated
num_convs times, then optionally MaxPool2d(2).

- Same padding (1) keeps spatial size constant through the convs; only the pool downsamples.
- in_channels → first conv; then each conv goes (hidden, hidden) until the last, which outputs out_channels
  (or all convs output the same out_channels for simplicity).
- Order: Conv → BN → ReLU (no ReLU after the last BN if you want to match some specs; here use ReLU after every conv).

Read: Popular_Architectures_Explained.md — “Why Stacks of 3×3”, “VGG-Style”.

Implement VGGBlock: __init__ builds the conv+BN+ReLU sequence and optional pool; forward runs them.
Output shape: (N, out_channels, H//2, W//2) if pool, else (N, out_channels, H, W).
"""

import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        use_pool: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("Build num_convs layers of Conv3x3(pad=1)->BN->ReLU; optionally MaxPool2d(2).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Run conv stack then pool if use_pool. Return (N, out_channels, H', W').")


if __name__ == "__main__":
    block = VGGBlock(3, 64, num_convs=2, use_pool=True)
    x = torch.randn(2, 3, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)
    block2 = VGGBlock(64, 128, num_convs=2, use_pool=False)
    out2 = block2(out)
    assert out2.shape == (2, 128, 8, 8)
    print("VGGBlock OK.")

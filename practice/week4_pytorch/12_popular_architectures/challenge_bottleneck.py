"""
Popular Architectures — Bottleneck Block (ResNet-50 style)
============================================================
Implement a bottleneck residual block: 1×1 conv (reduce channels) → BN → ReLU →
3×3 conv → BN → ReLU → 1×1 conv (expand back) → BN; then add shortcut and ReLU.

- Typical widths: in_c → 1×1 → mid_c (e.g. out_c//4) → 3×3 → mid_c → 1×1 → out_c.
  So the 3×3 runs on fewer channels (mid_c), saving compute.
- Shortcut: identity when stride==1 and in_c==out_c; else 1×1 conv (in_c→out_c, same stride) + BN.
- Stride is applied in the 3×3 conv (and in shortcut if present) so both paths match.

Read: Popular_Architectures_Explained.md — “Why Bottleneck Blocks”.

Implement BottleneckBlock(in_c, out_c, mid_c, stride=1). Output shape (N, out_c, H', W').
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, mid_c: int, stride: int = 1):
        super().__init__()
        raise NotImplementedError("1x1(in_c,mid_c), 3x3(mid_c,mid_c,stride), 1x1(mid_c,out_c); BN+ReLU; shortcut if needed.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Main path: 1x1->BN->ReLU, 3x3->BN->ReLU, 1x1->BN; add shortcut(x); ReLU.")

if __name__ == "__main__":
    block = BottleneckBlock(256, 256, 64, stride=1)
    x = torch.randn(2, 256, 8, 8)
    out = block(x)
    assert out.shape == (2, 256, 8, 8)
    block2 = BottleneckBlock(256, 512, 128, stride=2)
    out2 = block2(x)
    assert out2.shape == (2, 512, 4, 4)
    print("BottleneckBlock OK.")

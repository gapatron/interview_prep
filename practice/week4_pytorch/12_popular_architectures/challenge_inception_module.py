"""
Popular Architectures — Inception-Style Module (Parallel Branches, Concat)
==========================================================================
Implement a module that runs several conv branches in parallel and concatenates
their outputs along the channel dimension. All branches must output the same spatial size (H, W).

- Branch 1: 1×1 conv, in_c → c1.
- Branch 2: 1×1 (in_c → c2_in) → 3×3 (c2_in → c2), padding 1 so spatial same.
- Branch 3: 1×1 (in_c → c3_in) → 5×5 (c3_in → c3), padding 2 so spatial same.
- Branch 4: 3×3 MaxPool (padding 1, stride 1) → 1×1 conv (in_c → c4).
- Output: concat([b1, b2, b3, b4], dim=1) → (N, c1+c2+c3+c4, H, W).

Why: the network can learn to use different receptive fields (1×1, 3×3, 5×5, pool) in parallel.

Read: Popular_Architectures_Explained.md — “Inception-Style: Why Parallel Branches Then Concat”.

Implement InceptionModule(in_c, c1, c2_in, c2, c3_in, c3, c4). Out channels = c1+c2+c3+c4.
"""

import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_c: int,
        c1: int,
        c2_in: int,
        c2: int,
        c3_in: int,
        c3: int,
        c4: int,
    ):
        super().__init__()
        raise NotImplementedError("Four branches: 1x1; 1x1->3x3; 1x1->5x5; MaxPool3x3->1x1. Same H,W; concat on channels.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Run each branch, then torch.cat(..., dim=1). Return (N, c1+c2+c3+c4, H, W).")


if __name__ == "__main__":
    mod = InceptionModule(64, 16, 16, 32, 16, 16, 16)  # out = 16+32+16+16 = 80
    x = torch.randn(2, 64, 8, 8)
    out = mod(x)
    assert out.shape == (2, 80, 8, 8)
    print("InceptionModule OK.")

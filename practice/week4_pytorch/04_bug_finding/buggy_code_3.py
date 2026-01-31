"""
Bug 3 — Wrong dimension when conditioning a feature map
------------------------------------------------------
Concept: To concat cond (N,D) with x (N,C,H,W) along channels, cond must be (N,D,H,W). Use unsqueeze/expand, then cat(..., dim=1).
What goes wrong: Concatenating along the wrong dim (e.g. dim=0) gives the wrong shape. Run, find the bug, fix. Compare with solution_bug_3.py.
"""

import torch


def condition_to_feature_map(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """
    x: (N, C, H, W), cond: (N, D). Return (N, C+D, H, W).
    """
    N, C, H, W = x.shape
    D = cond.shape[1]
    cond = cond.unsqueeze(2).unsqueeze(3).expand(N, D, H, W)
    # BUG: cat along dim=0 stacks batches; we want new channels, so dim=1
    return torch.cat([x, cond], dim=0)


if __name__ == "__main__":
    x = torch.randn(2, 3, 4, 4)
    cond = torch.randn(2, 5)
    out = condition_to_feature_map(x, cond)
    print(out.shape)  # Expected (2, 8, 4, 4)

"""
Solution: Bug 3 — Conditioning feature map.
Fix: After expanding cond to (N,D,H,W), use torch.cat([x, cond], dim=1) so output is (N,C+D,H,W).
Why: dim=1 is the channel dimension; dim=0 would stack along batch.
"""

import torch


def condition_to_feature_map(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    D = cond.shape[1]
    cond = cond.unsqueeze(2).unsqueeze(3).expand(N, D, H, W)
    return torch.cat([x, cond], dim=1)

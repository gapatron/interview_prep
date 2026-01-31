"""
Solution: Bug 2 — Wrong cat dimension.
Fix: torch.cat([x_a, x_b], dim=1) so output is (N, in_a+in_b).
Why: dim=0 concatenates along batch (wrong shape); dim=1 concatenates along features.
"""

import torch
import torch.nn as nn


class ConcatBlock(nn.Module):
    def __init__(self, in_a: int, in_b: int, out: int):
        super().__init__()
        self.fc = nn.Linear(in_a + in_b, out)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_a, x_b], dim=1)
        return self.fc(x)

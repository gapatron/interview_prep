"""
Bug 2 — Wrong dimension in torch.cat
------------------------------------
Concept: cat(dim=0) stacks along batch; cat(dim=1) concatenates along features. For (N, in_a) and (N, in_b) we need (N, in_a+in_b).
What goes wrong: Concatenating along the wrong dim gives the wrong shape; the linear layer then fails or produces unexpected output.
Run, find the bug, fix. Compare with solution_bug_2.py.
"""

import torch
import torch.nn as nn


class ConcatBlock(nn.Module):
    def __init__(self, in_a: int, in_b: int, out: int):
        super().__init__()
        self.fc = nn.Linear(in_a + in_b, out)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_a, x_b], dim=0)
        return self.fc(x)


if __name__ == "__main__":
    block = ConcatBlock(10, 5, 8)
    x_a = torch.randn(4, 10)
    x_b = torch.randn(4, 5)
    out = block(x_a, x_b)
    print(out.shape)  # Expected (4, 8)

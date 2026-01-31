"""
Solution: ConcatBlock.
Takeaway: Linear input size = in_a + in_b; concat along dim=1 before the linear.
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


if __name__ == "__main__":
    block = ConcatBlock(10, 5, 8)
    x_a, x_b = torch.randn(4, 10), torch.randn(4, 5)
    assert block(x_a, x_b).shape == (4, 8)
    print("02 — ConcatBlock: OK.")

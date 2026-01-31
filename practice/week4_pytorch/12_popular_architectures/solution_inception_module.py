"""Solution: Inception module — four parallel branches (1x1, 1x1->3x3, 1x1->5x5, pool->1x1), concat on channels."""

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
        self.b1 = nn.Conv2d(in_c, c1, 1)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, c2_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2_in, c2, 3, padding=1),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, c3_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3_in, c3, 5, padding=2),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_c, c4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        return torch.cat([o1, o2, o3, o4], dim=1)


if __name__ == "__main__":
    mod = InceptionModule(64, 16, 16, 32, 16, 16, 16)
    x = torch.randn(2, 64, 8, 8)
    out = mod(x)
    assert out.shape == (2, 80, 8, 8)
    print("InceptionModule OK.")

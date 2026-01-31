"""Solution: Residual block with optional 1x1 shortcut for stride / channel change."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


if __name__ == "__main__":
    block = ResidualBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)
    block2 = ResidualBlock(64, 128, stride=2)
    out2 = block2(x)
    assert out2.shape == (2, 128, 4, 4)
    print("ResidualBlock OK.")

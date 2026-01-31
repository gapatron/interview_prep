"""Solution: Bottleneck block — 1x1(reduce) → 3x3 → 1x1(expand) + shortcut, ReLU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, mid_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, 1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(mid_c, out_c, 1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


if __name__ == "__main__":
    block = BottleneckBlock(256, 256, 64, stride=1)
    x = torch.randn(2, 256, 8, 8)
    out = block(x)
    assert out.shape == (2, 256, 8, 8)
    block2 = BottleneckBlock(256, 512, 128, stride=2)
    out2 = block2(x)
    assert out2.shape == (2, 512, 4, 4)
    print("BottleneckBlock OK.")

"""Solution: Simple CNN — stem + blocks + global pool + linear."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = SimpleCNN(3, 10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    print("SimpleCNN OK.")

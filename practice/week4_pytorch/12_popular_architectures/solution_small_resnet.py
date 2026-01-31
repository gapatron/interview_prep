"""Solution: Small ResNet — stem → layer1 → layer2 → layer3 → global pool → fc."""

import torch
import torch.nn as nn
from solution_make_layer import make_layer


class SmallResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = make_layer(128, 256, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = SmallResNet(10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    print("SmallResNet OK.")

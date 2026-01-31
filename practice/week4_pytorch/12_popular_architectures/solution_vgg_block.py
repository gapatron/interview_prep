"""Solution: VGG-style block — num_convs × (Conv3x3 pad1 → BN → ReLU), then optional MaxPool2d(2)."""

import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        use_pool: bool = True,
    ):
        super().__init__()
        layers = []
        for i in range(num_convs):
            inc = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv2d(inc, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return self.pool(x)


if __name__ == "__main__":
    block = VGGBlock(3, 64, num_convs=2, use_pool=True)
    x = torch.randn(2, 3, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)
    print("VGGBlock OK.")

"""Solution: make_layer — first block (in_c→out_c, stride), rest (out_c→out_c, stride=1)."""

import torch.nn as nn
from solution_residual_block import ResidualBlock


def make_layer(
    in_c: int,
    out_c: int,
    num_blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    blocks = [ResidualBlock(in_c, out_c, stride=stride)]
    for _ in range(num_blocks - 1):
        blocks.append(ResidualBlock(out_c, out_c, stride=1))
    return nn.Sequential(*blocks)


if __name__ == "__main__":
    import torch
    layer = make_layer(64, 128, num_blocks=2, stride=2)
    x = torch.randn(2, 64, 8, 8)
    out = layer(x)
    assert out.shape == (2, 128, 4, 4)
    print("make_layer OK.")

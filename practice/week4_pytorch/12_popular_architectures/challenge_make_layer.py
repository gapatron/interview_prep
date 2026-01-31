"""
Popular Architectures — ResNet “Layer” (make_layer)
===================================================
Implement a function or module that builds a sequence of ResidualBlocks forming one “layer”:
the first block may use a given stride and change channels (in_c → out_c);
the following (num_blocks - 1) blocks use stride=1 and keep out_c → out_c.

- This is how ResNet groups blocks: layer1 (e.g. 64 ch, 2 blocks), layer2 (first block stride 2, 64→128, then more blocks at 128), etc.
- You need ResidualBlock from challenge_residual_block (implement that first, or use the same logic inline).
- First block: ResidualBlock(in_c, out_c, stride=stride).
- Rest: ResidualBlock(out_c, out_c, stride=1), repeated (num_blocks - 1) times.

Read: Popular_Architectures_Explained.md — “ResNet Layer: Why Group Blocks Into Layers”.

Implement make_layer(in_c, out_c, num_blocks, stride=1) returning an nn.Sequential of ResidualBlocks.
"""

import torch
import torch.nn as nn


def make_layer(
    in_c: int,
    out_c: int,
    num_blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    raise NotImplementedError("First block: ResidualBlock(in_c, out_c, stride). Rest: ResidualBlock(out_c, out_c, 1) x (num_blocks-1).")


if __name__ == "__main__":
    # You must implement make_layer and have ResidualBlock in scope (implement challenge_residual_block first, or import it).
    layer = make_layer(64, 128, num_blocks=2, stride=2)
    x = torch.randn(2, 64, 8, 8)
    out = layer(x)
    assert out.shape == (2, 128, 4, 4)
    print("make_layer OK.")

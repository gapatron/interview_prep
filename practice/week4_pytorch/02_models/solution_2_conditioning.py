"""
Solution: ConditionalMLP.
Takeaway: First layer maps in_dim → h; concat hidden with condition (dim=1); second layer maps h+c_dim → num_classes.
"""
import torch
import torch.nn as nn


class ConditionalMLP(nn.Module):
    def __init__(self, in_dim: int, c_dim: int, h: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h + c_dim, num_classes)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h_c = torch.cat([h, c], dim=1)
        return self.fc2(h_c)


if __name__ == "__main__":
    model = ConditionalMLP(20, 4, 32, 3)
    x, c = torch.randn(5, 20), torch.randn(5, 4)
    assert model(x, c).shape == (5, 3)
    print("02 — ConditionalMLP: OK.")

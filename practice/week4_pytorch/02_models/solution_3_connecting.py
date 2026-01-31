"""
Solution: Encoder + Decoder + FullModel.
Takeaway: Store encoder and decoder as attributes; forward(x) = self.decoder(self.encoder(x)).
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class Decoder(nn.Module):
    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class FullModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden)
        self.decoder = Decoder(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


if __name__ == "__main__":
    model = FullModel(10, 20, 2)
    assert model(torch.randn(4, 10)).shape == (4, 2)
    print("02 — Encoder/Decoder/FullModel: OK.")

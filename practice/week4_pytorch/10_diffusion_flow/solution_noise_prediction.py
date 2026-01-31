"""Solution: Noise prediction model with t embedding and small conv backbone."""

import torch
import torch.nn as nn


class NoisePredictionNet(nn.Module):
    def __init__(self, in_channels: int, t_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.t_dim = t_dim
        self.t_embed = nn.Embedding(1000, t_dim)
        self.t_proj = nn.Linear(t_dim, in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t)
        t_emb = self.t_proj(t_emb)
        t_emb = t_emb.view(t_emb.size(0), -1, 1, 1)
        x = x_t + t_emb
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoisePredictionNet(in_channels=3, t_dim=32, hidden=64).to(device)
    x_t = torch.randn(2, 3, 8, 8, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    pred_eps = model(x_t, t)
    assert pred_eps.shape == x_t.shape, f"Expected {x_t.shape}, got {pred_eps.shape}"
    print("NoisePredictionNet OK.")

"""
10 — Noise-prediction model (diffusion)
========================================
Learning goal: In DDPM-style diffusion the model takes noised input x_t and timestep t and predicts the noise ε. Timestep must be fed in (embedding or sinusoidal) so the model knows which noise level.

Implement:
  - Embed t (N,) long → (N, t_dim); project to (N, in_channels); reshape to (N, C, 1, 1) and add to x_t (or concat).
  - Backbone: e.g. two Conv2d layers (in_channels → hidden → in_channels), ReLU between. Output same shape as x_t: (N, C, H, W).

Run the assert to check output shape. See Advanced_Architectures_Guide (diffusion chapter) for the training step.
"""

import torch
import torch.nn as nn


class NoisePredictionNet(nn.Module):
    """Predict noise epsilon from x_t and timestep t. Output shape = x_t shape."""

    def __init__(self, in_channels: int, t_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.t_dim = t_dim
        self.t_embed = nn.Embedding(1000, t_dim)  # T=1000 timesteps
        # TODO: project t_embed to in_channels for adding to x_t: Linear(t_dim, in_channels)
        # TODO: backbone: input is x_t + t_emb broadcast -> (N, C, H, W). Use Conv2d(in_channels, hidden, 3, padding=1) -> ReLU -> Conv2d(hidden, in_channels, 3, padding=1)
        pass

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (N,) long
        # TODO: t_emb = self.t_embed(t)  # (N, t_dim)
        # TODO: t_emb = self.t_proj(t_emb)  # (N, in_channels) -> view (N, C, 1, 1)
        # TODO: x = x_t + t_emb
        # TODO: return backbone(x)  # (N, C, H, W)
        pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoisePredictionNet(in_channels=3, t_dim=32, hidden=64).to(device)
    x_t = torch.randn(2, 3, 8, 8, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    pred_eps = model(x_t, t)
    assert pred_eps.shape == x_t.shape, f"Expected {x_t.shape}, got {pred_eps.shape}"
    print("NoisePredictionNet OK.")

"""
Diffusion / Flow — DDPM Training Step
=====================================
Implement one training step for DDPM-style diffusion:
  1. Sample t ~ Uniform(0, T-1), epsilon ~ N(0, I).
  2. Compute alpha_bar_t (from a simple schedule, e.g. linear beta then alpha_bar = cumprod(1 - beta)).
  3. x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.  (broadcast alpha_bar to (N,1,1,1))
  4. pred_eps = model(x_t, t)
  5. loss = MSE(pred_eps, epsilon)
Return loss. Use fixed T=100 and linear schedule: beta_t = 0.0001 + (0.02 - 0.0001) * t / (T-1).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_alpha_bars(T: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02, device=None):
    """Return alpha_bar for t in 0..T-1. alpha_bar[t] = prod(1 - beta[0:t+1])."""
    t_ar = torch.arange(T, dtype=torch.float32, device=device)
    beta = beta_start + (beta_end - beta_start) * t_ar / max(T - 1, 1)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha_bar


def ddpm_train_step(
    model: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    One DDPM training step. x0: (N, C, H, W), t: (N,) long in [0, T).
    alpha_bar: (T,) tensor so alpha_bar[t] is used for each sample.
    Return scalar loss.
    """
    # TODO: Sample epsilon ~ N(0, I) same shape as x0, on device
    # TODO: Gather alpha_bar_t for each sample: (N,) then view (N, 1, 1, 1)
    # TODO: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
    # TODO: pred_eps = model(x_t, t)
    # TODO: loss = F.mse_loss(pred_eps, epsilon); return loss
    pass


class MinimalNoiseNet(nn.Module):
    """Minimal model for testing: x_t, t -> same shape as x_t."""
    def __init__(self, C=3):
        super().__init__()
        self.conv = nn.Conv2d(C, C, 1)
    def forward(self, x_t, t):
        return self.conv(x_t)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 100
    alpha_bar = get_alpha_bars(T, device=device)
    model = MinimalNoiseNet(3).to(device)
    x0 = torch.randn(4, 3, 8, 8, device=device)
    t = torch.randint(0, T, (4,), device=device)
    loss = ddpm_train_step(model, x0, t, alpha_bar, device)
    assert loss.dim() == 0 and loss.item() >= 0
    print("DDPM train step OK.")

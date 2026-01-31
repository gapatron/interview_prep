"""Solution: DDPM training step — sample epsilon, form x_t, predict, MSE loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_alpha_bars(T: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02, device=None):
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
    N = x0.size(0)
    epsilon = torch.randn_like(x0, device=device)
    alpha_bar_t = alpha_bar[t].view(N, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon
    pred_eps = model(x_t, t)
    loss = F.mse_loss(pred_eps, epsilon)
    return loss


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from solution_noise_prediction import NoisePredictionNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 100
    alpha_bar = get_alpha_bars(T, device=device)
    model = NoisePredictionNet(3, 32, 64).to(device)
    x0 = torch.randn(4, 3, 8, 8, device=device)
    t = torch.randint(0, T, (4,), device=device)
    loss = ddpm_train_step(model, x0, t, alpha_bar, device)
    assert loss.dim() == 0 and loss.item() >= 0
    print("DDPM train step OK.")

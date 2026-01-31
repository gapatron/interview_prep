"""Solution: Reshape alpha_bar_t to (N, 1, 1, 1) for correct broadcast with (N, C, H, W)."""

import torch


def get_alpha_bars(T=100, device=None):
    t_ar = torch.arange(T, dtype=torch.float32, device=device)
    beta = 0.0001 + (0.02 - 0.0001) * t_ar / max(T - 1, 1)
    alpha = 1.0 - beta
    return torch.cumprod(alpha, dim=0)


def make_x_t(x0, t, alpha_bar, epsilon, device):
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_abar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
    x_t = sqrt_abar * x0 + sqrt_one_minus * epsilon
    return x_t


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_bar = get_alpha_bars(100, device=device)
    x0 = torch.randn(4, 3, 8, 8, device=device)
    t = torch.randint(0, 100, (4,), device=device)
    eps = torch.randn_like(x0, device=device)
    x_t = make_x_t(x0, t, alpha_bar, eps, device)
    assert x_t.shape == x0.shape
    print("OK.")

"""
Solution: LoRALinear — wrap nn.Linear with low-rank A, B; freeze base; forward = base(x) + scale * (x @ A.T @ B.T).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = scale
        self.base.requires_grad_(False)
        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # (x @ A.T) @ B.T = x @ (A.T @ B.T) = x @ (B @ A).T  so we need (B @ A) for F.linear(x, weight)
        lora_weight = self.lora_B @ self.lora_A
        lora_out = F.linear(x, lora_weight)
        return base_out + self.scale * lora_out


if __name__ == "__main__":
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, rank=2, scale=1.0)
    x = torch.randn(3, 8)
    out = lora(x)
    assert out.shape == (3, 4)
    assert not any(p.requires_grad for p in base.parameters())
    assert all(p.requires_grad for p in [lora.lora_A, lora.lora_B])
    out.sum().backward()
    print("LoRALinear OK.")

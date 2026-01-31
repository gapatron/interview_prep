"""
Base model used for LoRA and adapter injection. Self-contained so 13_lora_adapters
does not depend on 11_transformers. Provides named Linear layers for injection.
"""

import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    """Simple MLP with named Linear layers: fc1, fc2, fc3. Used for LoRA injection."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BaseTransformerBlock(nn.Module):
    """
    Minimal transformer-like block (no attention, just named linears) for adapter injection.
    Simulates structure: norm -> 'attention' (two linears) -> residual -> norm -> ff (two linears) -> residual.
    Input/output: (B, L, d_model). Named linears: proj_in, proj_out, ff1, ff2.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # "Attention" path (simplified as two linears)
        residual = x
        x = self.norm1(x)
        x = self.proj_in(x)
        x = torch.relu(x)
        x = self.proj_out(x)
        x = self.dropout(x) + residual
        # FFN path
        residual = x
        x = self.norm2(x)
        x = self.ff1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(self.ff2(x)) + residual
        return x

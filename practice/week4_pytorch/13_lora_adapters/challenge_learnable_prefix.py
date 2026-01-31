"""
Learnable Prefix (Prefix Tuning style)
======================================
Prepend a fixed number of **learnable** prefix vectors to the sequence so that
downstream attention sees them. Input (B, L, d_model) -> output (B, prefix_len + L, d_model).

- Store a learnable parameter of shape (prefix_len, d_model).
- Forward: for each batch, prepend the prefix to x. So output = cat(prefix_broadcast_to_batch, x, dim=1).
  prefix_batch shape (B, prefix_len, d_model) e.g. via prefix.unsqueeze(0).expand(B, -1, -1).
- Used in prefix tuning: these extra tokens act as "virtual" K,V (or context) for attention.

Implement LearnablePrefix.__init__ and forward. No residual; just prepend. See LoRA_and_Adapters_Guide.md.
"""

import torch
import torch.nn as nn


class LearnablePrefix(nn.Module):
    def __init__(self, prefix_len: int, d_model: int):
        super().__init__()
        raise NotImplementedError(
            "Learnable parameter (prefix_len, d_model). Forward: prepend prefix to x -> (B, prefix_len+L, d_model)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("cat(prefix_batch, x, dim=1).")

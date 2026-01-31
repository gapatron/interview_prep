"""
06 — Multi-input classifier (image + metadata)
==============================================
Learning goal: Combine two input streams (e.g. image and metadata) by separate branches, then concat features and classify. Reuses ConcatBlock / backbone patterns from 01–02.

Implement:
  - Image branch: flatten (view(N,-1)), Linear → ReLU → (N, hidden_img).
  - Metadata branch: Linear(meta_dim, hidden_meta) → ReLU → (N, hidden_meta).
  - Concat [img_feat, meta_feat] along dim=1; Linear(hidden_img + hidden_meta, num_classes) → logits.

Run the assert to check output shape.
"""

import torch
import torch.nn as nn


class MultiInputClassifier(nn.Module):
    """Two inputs: image and metadata. Concat features then classify."""

    def __init__(
        self,
        img_c: int,
        img_h: int,
        img_w: int,
        meta_dim: int,
        hidden_img: int,
        hidden_meta: int,
        num_classes: int,
    ):
        super().__init__()
        # TODO: image branch: flatten size = img_c * img_h * img_w -> Linear -> hidden_img
        # TODO: meta branch: Linear(meta_dim, hidden_meta) + ReLU
        # TODO: head: Linear(hidden_img + hidden_meta, num_classes)
        pass

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # TODO: img_feat = relu(self.img_fc(image.view(N, -1)))
        # TODO: meta_feat = relu(self.meta_fc(metadata))
        # TODO: concat -> head
        pass


if __name__ == "__main__":
    model = MultiInputClassifier(3, 8, 8, 5, 64, 32, 4)
    img = torch.randn(2, 3, 8, 8)
    meta = torch.randn(2, 5)
    out = model(img, meta)
    assert out.shape == (2, 4), out.shape
    print("MultiInputClassifier OK.")

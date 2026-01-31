"""Solution: Multi-input classifier (image + metadata)."""

import torch
import torch.nn as nn


class MultiInputClassifier(nn.Module):
    def __init__(self, img_c, img_h, img_w, meta_dim, hidden_img, hidden_meta, num_classes):
        super().__init__()
        flat = img_c * img_h * img_w
        self.img_fc = nn.Linear(flat, hidden_img)
        self.meta_fc = nn.Linear(meta_dim, hidden_meta)
        self.head = nn.Linear(hidden_img + hidden_meta, num_classes)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        N = image.size(0)
        img_feat = torch.relu(self.img_fc(image.view(N, -1)))
        meta_feat = torch.relu(self.meta_fc(metadata))
        concat = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(concat)

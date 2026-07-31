"""
models/fusion/attention_blocks.py

Standard channel- and spatial-attention building blocks, reused (not
claimed as novel) inside the frozen Task- and Hierarchy-Aware Evidential
Fusion module.

PRIOR WORK
------------
- Hu, Shen, Sun. "Squeeze-and-Excitation Networks." CVPR 2018. (channel
  attention via global-average-pool -> MLP -> sigmoid gate)
- Woo, Park, Lee, Kweon. "CBAM: Convolutional Block Attention Module."
  ECCV 2018. (spatial attention via channel-pooled avg/max maps -> conv)

WHY THEY'RE STILL HERE
-------------------------
Per-branch recalibration ("which channels matter," "where in the image
matters") is legitimate, well-established practice applied BEFORE the
task-aware evidential gating described in task_hierarchy_fusion.py — it is
not part of this project's novelty claim and is cited as such everywhere
it's used (configs/config.yaml, research/design_rationale.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)
        s = torch.sigmoid(self.mlp(s)).view(b, c, 1, 1)
        return x * s


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * m, m  # mask returned too, useful for explainability overlay

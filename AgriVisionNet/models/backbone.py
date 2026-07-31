"""
models/backbone.py

Dual-branch backbone for AgriVisionNet.

Branch 1 (CNN, EfficientNet-B0):
    Convolutional inductive bias -> strong at LOCAL, high-frequency texture
    patterns. Disease lesions (leaf spots, rust pustules, powdery mildew
    speckling) are exactly this kind of local, translation-invariant texture
    cue, so a CNN branch is well suited to detecting the *presence and
    fine-grained texture* of a lesion.

Branch 2 (Transformer, Swin-Tiny):
    Shifted-window self-attention -> strong at GLOBAL context and long-range
    dependencies (e.g. relating lesion position to leaf vein structure, or
    distinguishing overlapping diseases whose diagnosis depends on the
    holistic leaf appearance rather than a single patch). Swin also keeps
    a hierarchical, CNN-like resolution pyramid, which makes its 7x7 final
    feature map spatially compatible with EfficientNet's, avoiding the need
    for reshaping heuristics used when fusing ViT (single-scale) features.

Both branches are truncated at their final stage BEFORE global pooling, so
we keep a spatial feature map from each (not just a pooled vector). This is
required for the fusion module: cross-attention and spatial-attention need
spatial tokens/pixels to attend over, not a collapsed embedding.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    """EfficientNet-B0 feature extractor, returns the last conv feature map."""

    def __init__(self, name: str = "efficientnet_b0", pretrained: bool = True,
                 freeze_stem: bool = False):
        super().__init__()
        # features_only=True gives a list of feature maps at each stage
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]  # 1280 for B0

        if freeze_stem:
            for p in list(self.backbone.parameters())[:20]:
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feature map (B, C_cnn, h, w) — e.g. (B, 1280, 7, 7) at 224 input
        """
        feats = self.backbone(x)
        return feats[-1]


class SwinBranch(nn.Module):
    """Swin-Tiny feature extractor, returns the last-stage token grid reshaped to NCHW."""

    def __init__(self, name: str = "swin_tiny_patch4_window7_224", pretrained: bool = True,
                 freeze_patch_embed: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(3,)
        )
        self.out_channels = self.backbone.feature_info.channels()[-1]  # 768 for Swin-T

        if freeze_patch_embed:
            for n, p in self.backbone.named_parameters():
                if "patch_embed" in n:
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feature map (B, C_swin, h, w) — e.g. (B, 768, 7, 7) at 224 input

        timm's Swin with features_only returns (B, H, W, C) for the last
        stages in newer versions, or (B, N, C) tokens in older ones. We
        normalize both cases to a channel-first (B, C, H, W) map so the
        fusion module can treat both branches identically.
        """
        feats = self.backbone(x)[-1]

        if feats.dim() == 4 and feats.shape[1] != self.out_channels:
            # (B, H, W, C) -> (B, C, H, W)
            feats = feats.permute(0, 3, 1, 2).contiguous()
        elif feats.dim() == 3:
            # (B, N, C) -> (B, C, H, W), assume square token grid
            b, n, c = feats.shape
            h = w = int(n ** 0.5)
            feats = feats.transpose(1, 2).reshape(b, c, h, w).contiguous()

        return feats


class DualBranchBackbone(nn.Module):
    """Wraps both branches and exposes their (possibly different) spatial
    resolutions/channel counts. Resolution alignment (if needed) is handled
    inside the fusion module, not here, to keep this class a pure feature
    extractor."""

    def __init__(self, cfg: dict):
        super().__init__()
        cnn_cfg = cfg["backbone"]["cnn"]
        swin_cfg = cfg["backbone"]["transformer"]

        self.cnn_branch = CNNBranch(
            name=cnn_cfg["name"],
            pretrained=cnn_cfg["pretrained"],
            freeze_stem=cnn_cfg.get("freeze_stem", False),
        )
        self.swin_branch = SwinBranch(
            name=swin_cfg["name"],
            pretrained=swin_cfg["pretrained"],
            freeze_patch_embed=swin_cfg.get("freeze_patch_embed", False),
        )

    @property
    def cnn_channels(self) -> int:
        return self.cnn_branch.out_channels

    @property
    def swin_channels(self) -> int:
        return self.swin_branch.out_channels

    def forward(self, x: torch.Tensor):
        cnn_feat = self.cnn_branch(x)     # (B, C_cnn, h1, w1)
        swin_feat = self.swin_branch(x)   # (B, C_swin, h2, w2)
        return cnn_feat, swin_feat

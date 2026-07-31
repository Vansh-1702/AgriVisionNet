"""
datasets/transforms.py

Phase 2 objectives 6+7: Albumentations-based augmentation pipeline,
fully configurable through configs/config.yaml's `data.augmentation` block
(Phase 1's config already reserved this block -- no config schema change
needed here).

IMPORTANT SCOPE NOTE ON MIXUP/CUTMIX: these are NOT per-sample Albumentations
transforms -- they mix TWO samples together (image and label), which an
Albumentations `Compose` (one-sample-in, one-sample-out) cannot express.
The correct place for them is the training loop's collate step (Phase 3),
operating on an already-batched tensor. This module therefore provides:
  (a) the per-sample Albumentations pipeline (resize/normalize/flip/rotate/
      crop/color-jitter) -- used inside BaseCropDataset.__getitem__, and
  (b) the MixUp/CutMix LAMBDA-SAMPLING math and box-generation logic,
      framework-agnostic (pure numpy, no torch dependency, so it is
      actually exercised in this sandbox) -- Phase 3's collate_fn calls
      these functions and does the actual tensor mixing with torch.
Shipping (b) here now, ahead of Phase 3, avoids re-deriving the same beta-
distribution math in the training loop later and lets it be unit-tested
independently of a training run.

Albumentations itself is NOT installed in this authoring sandbox (no
network access to pip install it here) -- `build_train_transform()` /
`build_eval_transform()` are therefore UNVERIFIED in this environment; the
import is deferred (done inside the function, not at module load time) so
that importing this module for the MixUp/CutMix functions doesn't require
albumentations to be installed at all.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def build_train_transform(cfg: dict):
    """Build the training-time Albumentations pipeline from
    `configs/config.yaml`'s `data` block. Deferred import -- see module
    docstring re: albumentations not being installed in this sandbox.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    img_size = cfg["data"]["image_size"]
    aug = cfg["data"]["augmentation"]

    transforms = [A.Resize(img_size, img_size)]

    if aug.get("random_crop", False):
        # resize slightly larger first, then random-crop to target size --
        # a plain RandomCrop without prior upscaling would only ever crop
        # DOWN from whatever size Resize already fixed to img_size, i.e.
        # never actually do anything. This ordering is required, not stylistic.
        transforms = [A.Resize(int(img_size * 1.15), int(img_size * 1.15)),
                      A.RandomCrop(img_size, img_size)]

    if aug.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))

    rotation_degrees = aug.get("rotation_degrees", 0)
    if rotation_degrees:
        transforms.append(A.Rotate(limit=rotation_degrees, p=0.5))

    if aug.get("color_jitter", False):
        transforms.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5))

    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_eval_transform(cfg: dict):
    """Deterministic val/test pipeline -- resize + normalize only, no
    stochastic augmentation. Using the SAME resize target as training but
    none of the random ops is required for a fair evaluation; accidentally
    leaving an augmentation active here would silently make val/test
    metrics not comparable to a deployed inference pipeline."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    img_size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# -----------------------------------------------------------------------------
# MixUp / CutMix -- framework-agnostic math, actually testable without torch
# -----------------------------------------------------------------------------

def sample_mixup_lambda(alpha: float, rng: np.random.Generator) -> float:
    """Standard MixUp (Zhang et al., ICLR 2018) mixing coefficient.
    alpha<=0 is treated as "no mixing" (lambda=1.0) rather than raising,
    since a caller sweeping alpha down to 0 as an ablation shouldn't need
    a special-cased branch."""
    if alpha <= 0:
        return 1.0
    return float(rng.beta(alpha, alpha))


def sample_cutmix_box(image_size: Tuple[int, int], lam: float, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    """Standard CutMix (Yun et al., ICCV 2019) box sampling.

    Args:
        image_size: (H, W)
        lam: mixing coefficient (from sample_mixup_lambda or a separate
            CutMix-specific beta draw -- CutMix conventionally reuses the
            same lambda-sampling distribution as MixUp).
    Returns:
        (x1, y1, x2, y2) box coordinates to cut-and-paste from the second image.
    """
    H, W = image_size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = rng.integers(0, H)
    cx = rng.integers(0, W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    return int(x1), int(y1), int(x2), int(y2)


def cutmix_adjusted_lambda(box: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> float:
    """CutMix's actual mixing ratio is the realized cut-box area, NOT the
    lambda originally sampled -- clipping the box to image bounds (see
    sample_cutmix_box) changes the effective area, so the label-mixing
    coefficient used for the loss must be recomputed from the final box,
    not the pre-clip lambda. Using the pre-clip lambda for a clipped box
    is a real, easy-to-miss bug (silently mislabels the mixed sample's
    supervision target near image edges) -- documented here so Phase 3's
    collate_fn doesn't reintroduce it.
    """
    x1, y1, x2, y2 = box
    H, W = image_size
    box_area = (x2 - x1) * (y2 - y1)
    return 1.0 - (box_area / (H * W))

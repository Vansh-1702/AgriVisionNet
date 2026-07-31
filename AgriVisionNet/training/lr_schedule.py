"""
training/lr_schedule.py

Cosine-with-linear-warmup LR multiplier, matching
configs/config.yaml's `training.scheduler: cosine_warmup` +
`training.warmup_epochs`. Framework-agnostic (pure math, returns a
multiplier in [0,1] to scale the base LR) so it's testable here without
torch, and so training/trainer.py's actual `torch.optim.lr_scheduler.
LambdaLR(optimizer, lr_lambda=...)` (Phase 3, uses torch) can wrap this
function directly rather than reimplementing the schedule shape.
"""

from __future__ import annotations

import math


def cosine_warmup_multiplier(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.0,
) -> float:
    """
    Args:
        epoch: current epoch, 0-indexed.
        total_epochs: total planned training epochs.
        warmup_epochs: number of initial epochs to linearly ramp from 0 to 1.
        min_lr_ratio: floor for the multiplier at the end of cosine decay
            (0.0 = decays fully to 0; e.g. 0.01 keeps a small residual LR).
    Returns:
        multiplier in [min_lr_ratio, 1.0] -- multiply the base LR by this.
    """
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be > 0, got {total_epochs}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if warmup_epochs >= total_epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be < total_epochs "
            f"({total_epochs}) -- a schedule with no post-warmup decay "
            f"phase is very likely a config error, not intentional."
        )

    if epoch < warmup_epochs:
        # linear warmup: 0 -> 1 over warmup_epochs. epoch=0 gives a nonzero
        # starting multiplier (1/warmup_epochs), not literally 0, since an
        # LR of exactly 0 for the first optimizer step is a wasted step.
        return (epoch + 1) / warmup_epochs

    # cosine decay from 1.0 -> min_lr_ratio over the remaining epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = min(progress, 1.0)  # clamp -- caller may call with epoch >= total_epochs
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine

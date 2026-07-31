"""
training/early_stopping.py

Pure-Python early stopping bookkeeping -- no torch dependency, so this is
fully exercised in this authoring sandbox, unlike most of Phase 3.

DESIGN NOTE: tracks the metric history explicitly (not just a running best +
counter) so a resumed run (checkpoint/resume, per Phase 3's objectives) can
reconstruct its early-stopping state exactly from a saved history list,
rather than trusting that a resumed counter is consistent with what
actually happened before the crash/pause.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class EarlyStopping:
    """
    Args:
        patience: number of epochs with no improvement (by `min_delta`)
            after which training should stop.
        mode: "min" (lower metric is better, e.g. val loss) or "max"
            (higher is better, e.g. val accuracy/F1).
        min_delta: minimum absolute change to count as an improvement --
            without this, floating-point noise could reset the patience
            counter indefinitely on a genuinely plateaued metric.
    """
    patience: int = 12
    mode: str = "min"
    min_delta: float = 1e-4

    history: List[float] = dataclasses.field(default_factory=list)
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_since_improvement: int = 0

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{self.mode}'")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        return value > (self.best_value + self.min_delta)

    def step(self, value: float, epoch: int) -> bool:
        """Call once per epoch with the monitored validation metric.
        Returns True if training should stop NOW (this epoch is the last)."""
        self.history.append(value)

        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        return self.epochs_since_improvement >= self.patience

    def state_dict(self) -> dict:
        """For checkpoint/resume -- see training/checkpoint.py."""
        return dataclasses.asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict) -> "EarlyStopping":
        obj = cls(patience=state["patience"], mode=state["mode"], min_delta=state["min_delta"])
        obj.history = list(state["history"])
        obj.best_value = state["best_value"]
        obj.best_epoch = state["best_epoch"]
        obj.epochs_since_improvement = state["epochs_since_improvement"]
        return obj

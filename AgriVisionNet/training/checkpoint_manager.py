"""
training/checkpoint_manager.py

Checkpoint BOOKKEEPING (which checkpoint is "best", filename conventions,
resolved-config traceability per research/reproducibility.md Section 8)
separated from actual tensor serialization (torch.save/torch.load), so the
bookkeeping logic is testable without torch in this authoring sandbox.
training/trainer.py (torch-dependent, written but unverified here) calls
this class and hands it the actual state_dict to save.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class CheckpointManager:
    """
    Args:
        run_dir: e.g. outputs/runs/<run-name>/ -- matches the
            resolved_config.yaml convention already committed to in
            research/reproducibility.md Section 8.
        monitor_mode: "min" or "max", same semantics as
            training/early_stopping.py -- kept as a SEPARATE instance from
            EarlyStopping deliberately: a run might want to keep the best
            checkpoint by val loss while early-stopping on val F1, and
            conflating the two would prevent that.
        keep_last_n: also retain the N most recent checkpoints regardless
            of metric, so a crashed run can resume from its latest state
            even if that epoch wasn't the best one.
    """
    run_dir: Path
    monitor_mode: str = "min"
    keep_last_n: int = 3

    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    saved_epochs: list = field(default_factory=list)  # epochs with a checkpoint currently on disk

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        if self.monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got '{self.monitor_mode}'")

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value < self.best_value if self.monitor_mode == "min" else value > self.best_value

    def checkpoint_filename(self, epoch: int) -> str:
        return f"checkpoint_epoch{epoch:04d}.pt"

    def best_checkpoint_filename(self) -> str:
        return "checkpoint_best.pt"

    def latest_checkpoint_filename(self) -> str:
        return "checkpoint_latest.pt"

    def decide(self, epoch: int, monitored_value: float) -> dict:
        """Call once per epoch (after computing val metrics), BEFORE
        actually writing any files. Returns a plan describing what
        training/trainer.py's caller should do -- this method has no
        filesystem side effects itself, keeping it pure and testable.

        Returns:
            {
                "save_latest": bool,        # always True -- resume needs the latest state
                "save_best": bool,          # True iff this epoch improves on best_value
                "epochs_to_delete": [...],  # old "last-N" checkpoints to prune
                "is_new_best": bool,
            }
        """
        is_new_best = self._is_better(monitored_value)
        if is_new_best:
            self.best_value = monitored_value
            self.best_epoch = epoch

        self.saved_epochs.append(epoch)
        epochs_to_delete = []
        while len(self.saved_epochs) > self.keep_last_n:
            epochs_to_delete.append(self.saved_epochs.pop(0))

        return {
            "save_latest": True,
            "save_best": is_new_best,
            "epochs_to_delete": epochs_to_delete,
            "is_new_best": is_new_best,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }

    def save_resolved_config(self, resolved_config: dict, writer: Optional[Callable] = None) -> Path:
        """Writes resolved_config.yaml per research/reproducibility.md
        Section 8's traceability requirement -- every figure/table must be
        traceable to the exact config that produced it. `writer` defaults
        to yaml.safe_dump if not given (kept injectable for testing without
        requiring pyyaml at import time, though pyyaml IS available in this
        sandbox and used by default)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.run_dir / "resolved_config.yaml"

        if writer is None:
            import yaml
            out_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False))
        else:
            writer(resolved_config, out_path)

        return out_path

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "saved_epochs": list(self.saved_epochs),
            "monitor_mode": self.monitor_mode,
            "keep_last_n": self.keep_last_n,
        }

    @classmethod
    def from_state_dict(cls, run_dir, state: dict) -> "CheckpointManager":
        obj = cls(run_dir=run_dir, monitor_mode=state["monitor_mode"], keep_last_n=state["keep_last_n"])
        obj.best_value = state["best_value"]
        obj.best_epoch = state["best_epoch"]
        obj.saved_epochs = list(state["saved_epochs"])
        return obj

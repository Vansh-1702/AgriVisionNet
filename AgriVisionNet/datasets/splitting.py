"""
datasets/splitting.py

Phase 2 objective 5: train/validation/test splitting, stratified by the
combined (crop, disease) label so rare classes don't vanish entirely from
one split, seeded for reproducibility, and PERSISTED to disk as an explicit
list of indices -- not just re-derivable by rerunning the same code.

WHY PERSIST SPLIT INDICES RATHER THAN RELY ON "same seed = same split":
research/reproducibility.md Section 6 already commits to this ("the
resulting split indices should be saved to disk"). Re-deriving the split
from a seed is fragile across sklearn versions, dependency updates, or a
future refactor of the scanning order -- a saved index file is the only
thing that survives all of those unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split

from datasets.base_dataset import Sample


@dataclass
class DatasetSplit:
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def to_dict(self) -> dict:
        return {
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "DatasetSplit":
        payload = json.loads(Path(path).read_text())
        return cls(**payload)


def stratified_split(
    samples: List[Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_class_count_for_stratification: int = 3,
) -> DatasetSplit:
    """
    Args:
        samples: full sample list (indices in the returned DatasetSplit
            refer to positions in THIS list -- caller is responsible for
            keeping the list order stable between split-computation and
            split-use time).
        train_ratio, val_ratio, test_ratio: must sum to 1.0 (checked).
        seed: passed to sklearn's `random_state` for reproducibility.
        min_class_count_for_stratification: a class with fewer members than
            this cannot be safely stratified across a 3-way split. Such
            samples are assigned entirely to TRAIN with an explicit warning
            -- NOT silently dropped, and NOT force-fit into a stratified
            split sklearn would reject outright.
    """
    ratios_sum = train_ratio + val_ratio + test_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratios_sum}")

    labels = [f"{s.crop_name}|{s.disease_name}" for s in samples]
    label_counts: dict = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    stratifiable_idx = [i for i, lbl in enumerate(labels) if label_counts[lbl] >= min_class_count_for_stratification]
    unstratifiable_idx = [i for i, lbl in enumerate(labels) if label_counts[lbl] < min_class_count_for_stratification]

    if unstratifiable_idx:
        import warnings
        rare_classes = sorted({labels[i] for i in unstratifiable_idx})
        warnings.warn(
            f"[splitting] {len(unstratifiable_idx)} sample(s) across "
            f"{len(rare_classes)} class(es) have fewer than "
            f"{min_class_count_for_stratification} members and cannot be "
            f"safely stratified across 3 splits: {rare_classes}. These are "
            f"assigned entirely to TRAIN (not dropped, not force-split) -- "
            f"the model will never see them at val/test time, which should "
            f"be reported as a limitation for any class this happens to."
        )

    if stratifiable_idx:
        strat_labels = [labels[i] for i in stratifiable_idx]

        # first split: train vs (val+test)
        train_idx_local, temp_idx_local = train_test_split(
            range(len(stratifiable_idx)),
            train_size=train_ratio,
            random_state=seed,
            stratify=strat_labels,
        )
        temp_labels = [strat_labels[i] for i in temp_idx_local]

        # second split: val vs test, proportioned within the remainder
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_idx_local, test_idx_local = train_test_split(
            temp_idx_local,
            train_size=relative_val_ratio,
            random_state=seed,
            stratify=temp_labels,
        )

        train_idx = [stratifiable_idx[i] for i in train_idx_local]
        val_idx = [stratifiable_idx[i] for i in val_idx_local]
        test_idx = [stratifiable_idx[i] for i in test_idx_local]
    else:
        train_idx, val_idx, test_idx = [], [], []

    train_idx = sorted(train_idx + unstratifiable_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)

    return DatasetSplit(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

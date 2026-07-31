"""tests/test_datasets/test_splitting.py"""

from __future__ import annotations

from pathlib import Path

import pytest

from datasets.base_dataset import Sample
from datasets.splitting import stratified_split, DatasetSplit


def _make_samples(n_classes=5, per_class=20, rare_class_size=0):
    samples = []
    for c in range(n_classes):
        for i in range(per_class):
            samples.append(
                Sample(filepath=Path(f"/fake/{c}_{i}.jpg"), crop_name="Apple",
                       disease_name=f"disease{c}", source_dataset="test")
            )
    for i in range(rare_class_size):
        samples.append(
            Sample(filepath=Path(f"/fake/rare_{i}.jpg"), crop_name="Apple",
                   disease_name="rare_disease", source_dataset="test")
        )
    return samples


def test_bad_ratios_raise():
    samples = _make_samples()
    with pytest.raises(ValueError):
        stratified_split(samples, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_splits_are_disjoint_and_complete():
    samples = _make_samples()
    split = stratified_split(samples, seed=42)
    s1, s2, s3 = set(split.train_indices), set(split.val_indices), set(split.test_indices)
    assert not (s1 & s2) and not (s1 & s3) and not (s2 & s3)
    assert len(s1) + len(s2) + len(s3) == len(samples)


def test_determinism_same_seed():
    samples = _make_samples()
    split1 = stratified_split(samples, seed=42)
    split2 = stratified_split(samples, seed=42)
    assert split1.train_indices == split2.train_indices
    assert split1.val_indices == split2.val_indices
    assert split1.test_indices == split2.test_indices


def test_different_seeds_generally_differ():
    samples = _make_samples()
    split1 = stratified_split(samples, seed=42)
    split2 = stratified_split(samples, seed=99)
    assert split1.train_indices != split2.train_indices


def test_rare_class_entirely_in_train_with_warning():
    samples = _make_samples(rare_class_size=2)
    with pytest.warns(UserWarning, match="cannot be safely stratified"):
        split = stratified_split(samples, seed=42)

    rare_positions = [i for i, s in enumerate(samples) if s.disease_name == "rare_disease"]
    train_set = set(split.train_indices)
    assert all(i in train_set for i in rare_positions)


def test_save_load_roundtrip(tmp_path):
    samples = _make_samples()
    split = stratified_split(samples, seed=42)
    path = tmp_path / "split.json"
    split.save(path)
    loaded = DatasetSplit.load(path)
    assert loaded.train_indices == split.train_indices
    assert loaded.seed == split.seed

"""tests/test_datasets/test_rice_disease.py"""

from __future__ import annotations

import pytest

from datasets.rice_disease import RiceDiseaseDataset


def test_layout_a_split_subfolders(rice_disease_root_layout_a):
    ds = RiceDiseaseDataset(root=rice_disease_root_layout_a)
    # 2 splits x 2 classes x 2 images = 8
    assert len(ds) == 8
    assert all(s.crop_name == "rice" for s in ds.samples)


def test_layout_b_no_split_subfolders(rice_disease_root_layout_b):
    ds = RiceDiseaseDataset(root=rice_disease_root_layout_b)
    # 2 classes x 2 images = 4
    assert len(ds) == 4
    assert all(s.crop_name == "rice" for s in ds.samples)


def test_healthy_normalization(rice_disease_root_layout_b):
    ds = RiceDiseaseDataset(root=rice_disease_root_layout_b)
    diseases = {s.disease_name for s in ds.samples}
    assert "healthy" in diseases  # "Healthy" folder -> normalized to "healthy"


def test_neither_layout_raises_clear_error(tmp_path):
    """Neither split-subfolders nor class-subfolders present -- the
    documented ambiguity means this really can happen with an unexpected
    real download, and must fail loudly, not silently return zero samples."""
    root = tmp_path / "weird_layout"
    root.mkdir()
    (root / "some_random_file.txt").write_text("not a dataset")
    with pytest.raises(ValueError, match="ambiguity"):
        RiceDiseaseDataset(root=root)

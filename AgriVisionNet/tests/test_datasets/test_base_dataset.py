"""
tests/test_datasets/test_base_dataset.py

Tests the BaseCropDataset contract itself (missing root, empty scan,
__getitem__'s no-torch behavior), independent of any concrete loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datasets.base_dataset import BaseCropDataset, Sample, MISSING_LABEL


class _TrivialDataset(BaseCropDataset):
    """Minimal concrete subclass for testing the base class in isolation."""
    name = "trivial"

    def __init__(self, root, transform=None, samples_to_return=None):
        self._samples_to_return = samples_to_return if samples_to_return is not None else []
        super().__init__(root, transform)

    def _scan(self):
        return self._samples_to_return


def test_missing_root_raises_filenotfounderror(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        _TrivialDataset(root=nonexistent)


def test_empty_scan_raises_valueerror(tmp_path):
    root = tmp_path / "exists_but_empty"
    root.mkdir()
    with pytest.raises(ValueError):
        _TrivialDataset(root=root, samples_to_return=[])


def test_missing_label_sentinel_is_negative_one():
    assert MISSING_LABEL == -1


def test_getitem_without_torch_raises_cleanly(tmp_path, monkeypatch):
    """Simulates the no-torch environment this project's own authoring
    sandbox runs in -- __getitem__ must fail loudly and clearly, not with
    an obscure AttributeError deep in some tensor call."""
    import datasets.base_dataset as bd
    monkeypatch.setattr(bd, "_HAS_TORCH", False)

    root = tmp_path / "root"
    root.mkdir()
    img_path = root / "fake.jpg"
    img_path.write_bytes(b"not a real image, never opened in this test")

    sample = Sample(filepath=img_path, crop_name="Apple", disease_name="scab", source_dataset="trivial")
    ds = _TrivialDataset(root=root, samples_to_return=[sample])

    with pytest.raises(RuntimeError, match="torch is required"):
        ds[0]

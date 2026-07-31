"""tests/test_datasets/test_paddy_doctor.py"""

from __future__ import annotations

import pytest

from datasets.paddy_doctor import PaddyDoctorDataset


def test_scan_finds_all_images(paddy_doctor_root):
    ds = PaddyDoctorDataset(root=paddy_doctor_root)
    assert len(ds) == 6  # 2 blast + 2 normal + 2 brown_spot, per conftest


def test_crop_is_always_rice(paddy_doctor_root):
    ds = PaddyDoctorDataset(root=paddy_doctor_root)
    assert all(s.crop_name == "rice" for s in ds.samples)


def test_normal_label_mapped_to_healthy(paddy_doctor_root):
    ds = PaddyDoctorDataset(root=paddy_doctor_root)
    diseases = {s.disease_name for s in ds.samples}
    assert "healthy" in diseases
    assert "normal" not in diseases  # must be remapped, not passed through raw


def test_missing_train_images_dir_raises(tmp_path):
    empty_root = tmp_path / "empty_paddy"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError):
        PaddyDoctorDataset(root=empty_root)


def test_severity_is_always_none(paddy_doctor_root):
    ds = PaddyDoctorDataset(root=paddy_doctor_root)
    assert all(s.severity_name is None for s in ds.samples)


def test_missing_metadata_csv_does_not_fail_scan(tmp_path):
    """train.csv is optional -- its absence must not break the core scan."""
    from PIL import Image
    root = tmp_path / "no_csv_paddy"
    img_dir = root / "train_images" / "blast"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(img_dir / "x.jpg")

    ds = PaddyDoctorDataset(root=root)  # no train.csv written
    assert len(ds) == 1

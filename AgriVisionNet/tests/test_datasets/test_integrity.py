"""tests/test_datasets/test_integrity.py"""

from __future__ import annotations

import shutil
from pathlib import Path

from datasets.integrity import run_integrity_checks, imbalance_severity, IMBALANCE_MILD, IMBALANCE_SEVERE
from datasets.plantvillage import PlantVillageDataset


def test_clean_dataset_reports_clean(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    report = run_integrity_checks(ds.samples, check_duplicates=True)
    assert report.is_clean
    assert report.corrupt_files == []
    assert report.zero_byte_files == []
    assert report.duplicate_groups == []


def test_detects_corrupt_file(plantvillage_root):
    corrupt_path = plantvillage_root / "Apple___Apple_scab" / "corrupt.jpg"
    corrupt_path.write_bytes(b"not a real jpeg")

    ds = PlantVillageDataset(root=plantvillage_root)
    report = run_integrity_checks(ds.samples)
    assert len(report.corrupt_files) == 1
    assert not report.is_clean


def test_detects_zero_byte_file(plantvillage_root):
    empty_path = plantvillage_root / "Apple___Apple_scab" / "empty.jpg"
    empty_path.write_bytes(b"")

    ds = PlantVillageDataset(root=plantvillage_root)
    report = run_integrity_checks(ds.samples)
    assert len(report.zero_byte_files) == 1


def test_detects_duplicate_content(plantvillage_root):
    src = plantvillage_root / "Apple___Apple_scab" / "img0.jpg"
    dup = plantvillage_root / "Apple___Apple_scab" / "img0_dup.jpg"
    shutil.copy(src, dup)

    ds = PlantVillageDataset(root=plantvillage_root)
    report = run_integrity_checks(ds.samples, check_duplicates=True)
    assert len(report.duplicate_groups) == 1
    assert len(report.duplicate_groups[0]) == 2


def test_missing_file_detected(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    victim = ds.samples[0].filepath
    victim.unlink()  # delete after scan, simulating a file removed post-index

    report = run_integrity_checks(ds.samples)
    assert len(report.missing_files) == 1


def test_class_counts_and_imbalance_ratio(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    report = run_integrity_checks(ds.samples, check_duplicates=False)
    assert report.class_counts["Apple|Apple_scab"] == 3
    assert report.class_counts["Apple|healthy"] == 2
    assert report.imbalance_ratio == 3 / 2


def test_imbalance_severity_thresholds():
    assert imbalance_severity(1.0) == "none"
    assert imbalance_severity(IMBALANCE_MILD) == "mild"
    assert imbalance_severity(IMBALANCE_SEVERE) == "severe"
    assert imbalance_severity(IMBALANCE_SEVERE + 5) == "severe"

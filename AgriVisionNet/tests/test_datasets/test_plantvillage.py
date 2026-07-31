"""tests/test_datasets/test_plantvillage.py"""

from __future__ import annotations

import pytest

from datasets.plantvillage import PlantVillageDataset, parse_plantvillage_folder_name


def test_folder_name_parsing():
    assert parse_plantvillage_folder_name("Apple___Apple_scab") == ("Apple", "Apple_scab")
    assert parse_plantvillage_folder_name("Corn_(maize)___Common_rust_") == ("Corn_(maize)", "Common_rust_")


def test_folder_name_parsing_rejects_missing_separator():
    with pytest.raises(ValueError):
        parse_plantvillage_folder_name("AppleScab")  # no '___' separator


def test_scan_finds_all_images(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    assert len(ds) == 3 + 2 + 2 + 2  # matches conftest's layout counts


def test_scan_extracts_correct_crop_and_disease(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    crops = {s.crop_name for s in ds.samples}
    diseases = {s.disease_name for s in ds.samples}
    assert crops == {"Apple", "Tomato", "Corn_(maize)"}
    assert "healthy" in diseases
    assert "Apple_scab" in diseases


def test_severity_is_always_none(plantvillage_root):
    """Documented data gap -- PlantVillage provides no severity labels."""
    ds = PlantVillageDataset(root=plantvillage_root)
    assert all(s.severity_name is None for s in ds.samples)


def test_source_dataset_tag_is_set(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    assert all(s.source_dataset == "plantvillage" for s in ds.samples)

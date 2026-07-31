"""
tests/test_datasets/test_unified.py

Covers UnifiedCropDataset's core correctness property: the same real-world
crop/disease, expressed with different raw strings across different
datasets (e.g. "rice" from both paddy_doctor and rice_disease), must
resolve to the SAME integer label. This is the property that makes
"add a new dataset with minimal code changes" actually safe, not just
convenient.
"""

from __future__ import annotations

import pytest

from datasets.unified import UnifiedCropDataset


@pytest.fixture
def multi_dataset_configs(plantvillage_root, paddy_doctor_root, rice_disease_root_layout_b):
    return [
        {"name": "plantvillage", "root": str(plantvillage_root), "enabled": True},
        {"name": "paddy_doctor", "root": str(paddy_doctor_root), "enabled": True},
        {"name": "rice_disease", "root": str(rice_disease_root_layout_b), "enabled": True},
    ]


def test_disabled_dataset_excluded(plantvillage_root, paddy_doctor_root, rice_disease_root_layout_b):
    configs = [
        {"name": "plantvillage", "root": str(plantvillage_root), "enabled": True},
        {"name": "paddy_doctor", "root": str(paddy_doctor_root), "enabled": False},
        {"name": "rice_disease", "root": str(rice_disease_root_layout_b), "enabled": False},
    ]
    ds = UnifiedCropDataset(configs)
    assert all(s.source_dataset == "plantvillage" for s in ds.all_samples())


def test_no_enabled_dataset_raises(plantvillage_root):
    configs = [{"name": "plantvillage", "root": str(plantvillage_root), "enabled": False}]
    with pytest.raises(ValueError, match="enabled=True"):
        UnifiedCropDataset(configs)


def test_cross_dataset_rice_gets_same_integer_id(multi_dataset_configs):
    ds = UnifiedCropDataset(multi_dataset_configs)

    rice_samples = [s for s in ds.all_samples() if s.crop_name == "rice"]
    sources = {s.source_dataset for s in rice_samples}
    assert sources == {"paddy_doctor", "rice_disease"}

    encoded_ids = {ds.crop_space.encode(s.crop_name) for s in rice_samples}
    assert len(encoded_ids) == 1  # same crop, same ID, regardless of source dataset


def test_severity_space_is_empty_given_documented_gap(multi_dataset_configs):
    """None of the three built-in loaders provide severity ground truth --
    this is a real data gap, and the unified severity label space should
    reflect that honestly (zero classes), not silently synthesize one."""
    ds = UnifiedCropDataset(multi_dataset_configs)
    assert ds.severity_space.num_classes == 0


def test_label_space_reuse_prevents_val_leakage(multi_dataset_configs):
    """A label space fitted on train must be reused unchanged for val/test
    -- refitting per-split would let val/test 'see' classes absent from
    train, which is encoding leakage."""
    train_ds = UnifiedCropDataset(multi_dataset_configs)
    spaces = train_ds.label_spaces()

    eval_ds = UnifiedCropDataset(multi_dataset_configs, label_spaces=spaces)
    assert eval_ds.crop_space is spaces["crop"]
    assert eval_ds.crop_space.num_classes == train_ds.crop_space.num_classes


def test_total_length_is_sum_of_enabled_subdatasets(multi_dataset_configs):
    ds = UnifiedCropDataset(multi_dataset_configs)
    from datasets.plantvillage import PlantVillageDataset
    from datasets.paddy_doctor import PaddyDoctorDataset
    from datasets.rice_disease import RiceDiseaseDataset

    expected = 0
    for cfg in multi_dataset_configs:
        cls = {"plantvillage": PlantVillageDataset, "paddy_doctor": PaddyDoctorDataset,
               "rice_disease": RiceDiseaseDataset}[cfg["name"]]
        expected += len(cls(root=cfg["root"]))

    assert len(ds) == expected

"""tests/test_datasets/test_statistics.py"""

from __future__ import annotations

from datasets.plantvillage import PlantVillageDataset
from datasets.statistics import compute_dataset_statistics


def test_total_and_severity_coverage(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    stats = compute_dataset_statistics(ds.samples)
    assert stats.total_samples == len(ds)
    assert stats.severity_coverage == {"labeled": 0, "missing": len(ds)}


def test_per_crop_counts(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    stats = compute_dataset_statistics(ds.samples)
    assert stats.per_crop_counts["Apple"] == 5  # 3 scab + 2 healthy, per conftest
    assert stats.per_crop_counts["Tomato"] == 2


def test_image_size_distribution(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    stats = compute_dataset_statistics(ds.samples, probe_image_sizes=True)
    assert stats.num_unique_image_sizes == 1  # all conftest fixtures are 16x16
    assert stats.image_size_distribution["16x16"] == len(ds)


def test_probe_disabled_skips_size_distribution(plantvillage_root):
    ds = PlantVillageDataset(root=plantvillage_root)
    stats = compute_dataset_statistics(ds.samples, probe_image_sizes=False)
    assert stats.image_size_distribution == {}


def test_save_produces_valid_json(plantvillage_root, tmp_path):
    import json
    ds = PlantVillageDataset(root=plantvillage_root)
    stats = compute_dataset_statistics(ds.samples)
    out_path = tmp_path / "stats.json"
    stats.save(out_path)
    loaded = json.loads(out_path.read_text())
    assert loaded["total_samples"] == len(ds)

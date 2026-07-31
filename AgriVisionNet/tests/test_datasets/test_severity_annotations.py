"""
tests/test_datasets/test_severity_annotations.py

All CSV data in these tests is a SYNTHETIC SCHEMA-TEST FIXTURE, never real
annotation data -- research/severity_annotation_protocol.md's actual
annotation process has not been run yet. These tests only verify the
loader/overlay/kappa code is correct against the documented CSV schema.
"""

from __future__ import annotations

import warnings

import pytest

from datasets.severity_annotations import (
    load_severity_annotations,
    attach_severity_labels,
    compute_interrater_kappa,
    interpret_kappa,
    REQUIRED_COLUMNS,
)


CSV_HEADER = ",".join(REQUIRED_COLUMNS)


@pytest.fixture
def severity_csv(tmp_path, plantvillage_root):
    """Builds a schema-test CSV keyed to the REAL fixture filepaths from
    conftest.py's plantvillage_root, covering: unanimous agreement, an
    adjudicated adjacent disagreement, an unresolved (blank) row, and a
    deliberate crop-name mismatch to test the mismatch-detection path."""
    from datasets.plantvillage import PlantVillageDataset

    ds = PlantVillageDataset(root=plantvillage_root)
    scab_paths = sorted(str(s.filepath) for s in ds.samples if s.disease_name == "Apple_scab")
    healthy_paths = sorted(str(s.filepath) for s in ds.samples if s.disease_name == "healthy")

    rows = [
        CSV_HEADER,
        f"{scab_paths[0]},plantvillage,Apple,Apple_scab,A1,mild,A2,mild,,mild,0,0,,unanimous",
        f"{scab_paths[1]},plantvillage,Apple,Apple_scab,A1,moderate,A2,severe,ADJ1,severe,0,0,,adjudicated",
        f"{scab_paths[2]},plantvillage,Apple,Apple_scab,A1,mild,A2,moderate,,,0,0,,unresolved blank on purpose",
        f"{healthy_paths[0]},plantvillage,Apple,healthy,A1,healthy,A2,healthy,,healthy,0,0,,healthy row",
        f"{healthy_paths[1]},plantvillage,WRONG_CROP,healthy,A1,healthy,A2,healthy,,healthy,0,0,,deliberate mismatch",
    ]
    path = tmp_path / "severity_test_fixture.csv"
    path.write_text("\n".join(rows) + "\n")
    return path, ds


def test_load_severity_annotations_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_severity_annotations(tmp_path / "does_not_exist.csv")


def test_load_severity_annotations_wrong_schema_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("wrong,columns\n1,2\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_severity_annotations(bad_csv)


def test_load_severity_annotations_counts(severity_csv):
    path, _ds = severity_csv
    store = load_severity_annotations(path)
    assert store.num_rows == 5
    assert store.resolved_count() == 4  # 4 rows have a valid resolved_severity in the CSV; 1 blank


def test_attach_severity_labels_correct_values(severity_csv):
    path, ds = severity_csv
    store = load_severity_annotations(path)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        attached = attach_severity_labels(ds.samples, store)
        assert len(w) == 1
        assert "mismatch" in str(w[0].message)

    by_path = {str(s.filepath): s for s in attached}
    scab_paths = sorted(p for p in by_path if "Apple_scab" in p)
    healthy_paths = sorted(p for p in by_path if "healthy" in p)

    assert by_path[scab_paths[0]].severity_name == "mild"
    assert by_path[scab_paths[1]].severity_name == "severe"
    assert by_path[scab_paths[2]].severity_name is None  # blank resolved_severity
    assert by_path[healthy_paths[0]].severity_name == "healthy"
    assert by_path[healthy_paths[1]].severity_name is None  # crop mismatch -- NOT attached


def test_unmatched_sample_keeps_none(severity_csv, tmp_path):
    """A Sample whose filepath isn't in the CSV at all must be untouched."""
    from datasets.base_dataset import Sample
    from pathlib import Path

    path, _ds = severity_csv
    store = load_severity_annotations(path)
    unrelated = Sample(filepath=Path("/nowhere/x.jpg"), crop_name="Apple",
                        disease_name="Apple_scab", source_dataset="plantvillage")
    result = attach_severity_labels([unrelated], store)
    assert result[0].severity_name is None


def test_compute_interrater_kappa(severity_csv):
    path, _ds = severity_csv
    store = load_severity_annotations(path)
    result = compute_interrater_kappa(store)
    assert result["n_pairs"] == 5
    assert -1.0 <= result["kappa"] <= 1.0
    assert result["interpretation"] in {"slight", "fair", "moderate", "substantial", "almost perfect"}


def test_kappa_requires_minimum_pairs(tmp_path):
    csv_path = tmp_path / "too_few.csv"
    csv_path.write_text(
        CSV_HEADER + "\n"
        "/a.jpg,plantvillage,Apple,scab,A1,mild,A2,,,,0,0,,incomplete row\n"
    )
    store = load_severity_annotations(csv_path)
    with pytest.raises(ValueError, match="at least 2"):
        compute_interrater_kappa(store)


@pytest.mark.parametrize("kappa,expected", [
    (0.10, "slight"),
    (0.30, "fair"),
    (0.50, "moderate"),
    (0.70, "substantial"),
    (0.90, "almost perfect"),
])
def test_interpret_kappa_bands(kappa, expected):
    assert interpret_kappa(kappa) == expected

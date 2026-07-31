"""tests/test_datasets/test_label_space.py"""

from __future__ import annotations

import pytest

from datasets.base_dataset import MISSING_LABEL
from datasets.label_space import LabelSpace, normalize_label


def test_normalize_label_collision():
    assert normalize_label("Corn_(maize)") == normalize_label("corn maize") == "corn maize"
    assert normalize_label("Apple_scab") == "apple scab"


def test_fit_collapses_normalized_duplicates():
    ls = LabelSpace().fit(["Apple", "rice", "Apple", "Corn_(maize)", "corn maize"])
    assert ls.num_classes == 3  # not 4 -- normalization collision


def test_encode_decode_roundtrip():
    ls = LabelSpace().fit(["Apple", "rice", "Tomato"])
    idx = ls.encode("rice")
    assert ls.decode(idx) == "rice"


def test_cross_form_encode_gives_same_id():
    ls = LabelSpace().fit(["Corn_(maize)"])
    assert ls.encode("Corn_(maize)") == ls.encode("corn maize")


def test_missing_label_sentinel():
    ls = LabelSpace().fit(["Apple"])
    assert ls.encode(None) == MISSING_LABEL == -1


def test_unseen_label_raises_keyerror():
    ls = LabelSpace().fit(["Apple"])
    with pytest.raises(KeyError):
        ls.encode("banana")


def test_encode_before_fit_raises():
    ls = LabelSpace()
    with pytest.raises(RuntimeError):
        ls.encode("Apple")


def test_save_load_roundtrip(tmp_path):
    ls = LabelSpace().fit(["Apple", "rice", "Tomato"])
    path = tmp_path / "labelspace.json"
    ls.save(path)
    loaded = LabelSpace.load(path)
    assert loaded.num_classes == ls.num_classes
    assert loaded.encode("rice") == ls.encode("rice")


def test_fit_is_deterministic():
    ls1 = LabelSpace().fit(["Tomato", "Apple", "rice"])
    ls2 = LabelSpace().fit(["rice", "Tomato", "Apple"])  # different input order
    assert ls1.encode("Apple") == ls2.encode("Apple")
    assert ls1.encode("rice") == ls2.encode("rice")

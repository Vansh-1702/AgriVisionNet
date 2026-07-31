"""
tests/test_datasets/test_transforms.py

Only covers the torch-free MixUp/CutMix math (sample_mixup_lambda,
sample_cutmix_box, cutmix_adjusted_lambda). build_train_transform() /
build_eval_transform() require `albumentations`, which is NOT installed in
this authoring sandbox (see transforms.py's module docstring) -- those two
functions are therefore UNTESTED here; test them first, before Phase 3
training, once albumentations is actually installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.transforms import sample_mixup_lambda, sample_cutmix_box, cutmix_adjusted_lambda


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_mixup_lambda_no_mixing_for_nonpositive_alpha(rng):
    assert sample_mixup_lambda(0.0, rng) == 1.0
    assert sample_mixup_lambda(-1.0, rng) == 1.0


def test_mixup_lambda_within_bounds(rng):
    lams = [sample_mixup_lambda(0.2, rng) for _ in range(500)]
    assert all(0.0 <= l <= 1.0 for l in lams)


def test_cutmix_box_within_image_bounds(rng):
    H, W = 224, 224
    for lam in [0.05, 0.5, 0.95]:
        x1, y1, x2, y2 = sample_cutmix_box((H, W), lam, rng)
        assert 0 <= x1 <= x2 <= W
        assert 0 <= y1 <= y2 <= H


def test_cutmix_adjusted_lambda_matches_area_formula():
    box = (0, 0, 100, 100)
    adj = cutmix_adjusted_lambda(box, (224, 224))
    expected = 1.0 - (100 * 100) / (224 * 224)
    assert abs(adj - expected) < 1e-9


def test_cutmix_adjusted_lambda_zero_area_box_gives_lambda_one():
    box = (50, 50, 50, 50)
    assert cutmix_adjusted_lambda(box, (224, 224)) == 1.0


def test_cutmix_adjusted_lambda_full_image_box_gives_lambda_zero():
    box = (0, 0, 224, 224)
    assert cutmix_adjusted_lambda(box, (224, 224)) == 0.0

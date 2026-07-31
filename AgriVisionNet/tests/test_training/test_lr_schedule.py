"""tests/test_training/test_lr_schedule.py"""

from __future__ import annotations

import pytest

from training.lr_schedule import cosine_warmup_multiplier


def test_warmup_phase_monotonic_increasing():
    total, warmup = 100, 5
    vals = [cosine_warmup_multiplier(e, total, warmup) for e in range(warmup)]
    assert vals == sorted(vals)
    assert vals[0] == pytest.approx(1 / warmup)
    assert vals[-1] == pytest.approx(1.0)


def test_cosine_phase_monotonic_decreasing():
    total, warmup = 100, 5
    vals = [cosine_warmup_multiplier(e, total, warmup) for e in range(warmup, total)]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


def test_start_and_end_of_cosine_phase():
    total, warmup = 100, 5
    assert cosine_warmup_multiplier(warmup, total, warmup) > 0.99
    assert cosine_warmup_multiplier(total - 1, total, warmup) < 0.01


def test_min_lr_ratio_floor():
    total, warmup = 100, 5
    v = cosine_warmup_multiplier(total - 1, total, warmup, min_lr_ratio=0.1)
    assert v >= 0.1 - 1e-9


def test_clamped_beyond_total_epochs():
    v = cosine_warmup_multiplier(150, 100, 5)
    assert v == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("kwargs", [
    dict(epoch=0, total_epochs=0, warmup_epochs=0),
    dict(epoch=0, total_epochs=10, warmup_epochs=-1),
    dict(epoch=0, total_epochs=10, warmup_epochs=10),
])
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        cosine_warmup_multiplier(**kwargs)

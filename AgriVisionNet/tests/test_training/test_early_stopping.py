"""tests/test_training/test_early_stopping.py"""

from __future__ import annotations

import pytest

from training.early_stopping import EarlyStopping


def test_mode_min_stops_after_patience_epochs_no_improvement():
    es = EarlyStopping(patience=3, mode="min", min_delta=0.001)
    losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]
    stop_epoch = None
    for epoch, loss in enumerate(losses):
        if es.step(loss, epoch):
            stop_epoch = epoch
            break
    assert stop_epoch == 5
    assert es.best_epoch == 2
    assert es.best_value == 0.85


def test_mode_max_stops_after_patience_epochs_no_improvement():
    es = EarlyStopping(patience=2, mode="max", min_delta=0.01)
    accs = [0.5, 0.6, 0.65, 0.64, 0.63]
    stop = False
    for epoch, acc in enumerate(accs):
        stop = es.step(acc, epoch)
    assert es.best_value == 0.65
    assert es.best_epoch == 2
    assert stop is True


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        EarlyStopping(mode="bogus")


def test_invalid_patience_raises():
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_state_dict_roundtrip():
    es = EarlyStopping(patience=3, mode="min")
    for epoch, loss in enumerate([1.0, 0.9, 0.95, 0.96]):
        es.step(loss, epoch)

    state = es.state_dict()
    resumed = EarlyStopping.from_state_dict(state)
    assert resumed.best_value == es.best_value
    assert resumed.best_epoch == es.best_epoch
    assert resumed.history == es.history
    assert resumed.epochs_since_improvement == es.epochs_since_improvement


def test_min_delta_prevents_noise_resetting_patience():
    """A tiny 'improvement' smaller than min_delta must NOT reset the
    patience counter -- otherwise floating-point noise on a genuinely
    plateaued metric would prevent early stopping from ever triggering."""
    es = EarlyStopping(patience=2, mode="min", min_delta=0.01)
    es.step(1.0, 0)
    es.step(0.999, 1)  # "improvement" of 0.001 < min_delta=0.01 -- should NOT count
    assert es.epochs_since_improvement == 1
    stop = es.step(0.998, 2)
    assert es.epochs_since_improvement == 2
    assert stop is True

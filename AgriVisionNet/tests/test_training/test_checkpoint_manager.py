"""tests/test_training/test_checkpoint_manager.py"""

from __future__ import annotations

import pytest
import yaml

from training.checkpoint_manager import CheckpointManager


def test_decide_sequence_tracks_best_and_prunes(tmp_path):
    cm = CheckpointManager(run_dir=tmp_path / "run1", monitor_mode="min", keep_last_n=2)

    plan0 = cm.decide(0, 1.0)
    assert plan0["is_new_best"] is True
    assert plan0["epochs_to_delete"] == []

    plan1 = cm.decide(1, 0.8)
    assert plan1["is_new_best"] is True
    assert plan1["epochs_to_delete"] == []

    plan2 = cm.decide(2, 0.9)  # worse than best (0.8), but keep_last_n=2 prunes epoch 0
    assert plan2["is_new_best"] is False
    assert plan2["epochs_to_delete"] == [0]
    assert plan2["best_epoch"] == 1
    assert plan2["best_value"] == 0.8

    plan3 = cm.decide(3, 0.5)
    assert plan3["is_new_best"] is True
    assert plan3["epochs_to_delete"] == [1]
    assert cm.best_epoch == 3


def test_monitor_mode_max():
    cm = CheckpointManager(run_dir="/tmp/x", monitor_mode="max", keep_last_n=5)
    cm.decide(0, 0.5)
    cm.decide(1, 0.7)
    plan = cm.decide(2, 0.6)
    assert plan["is_new_best"] is False
    assert cm.best_value == 0.7
    assert cm.best_epoch == 1


def test_invalid_monitor_mode_raises():
    with pytest.raises(ValueError):
        CheckpointManager(run_dir="/tmp/x", monitor_mode="bogus")


def test_save_resolved_config_traceability(tmp_path):
    cm = CheckpointManager(run_dir=tmp_path / "run1")
    resolved = {"training": {"lr": 0.001}, "seed": 42}
    out_path = cm.save_resolved_config(resolved)
    assert out_path.exists()
    loaded = yaml.safe_load(out_path.read_text())
    assert loaded == resolved


def test_state_dict_roundtrip(tmp_path):
    cm = CheckpointManager(run_dir=tmp_path / "run1", monitor_mode="min", keep_last_n=2)
    cm.decide(0, 1.0)
    cm.decide(1, 0.8)
    cm.decide(2, 0.9)

    state = cm.state_dict()
    resumed = CheckpointManager.from_state_dict(tmp_path / "run1", state)
    assert resumed.best_value == cm.best_value
    assert resumed.best_epoch == cm.best_epoch
    assert resumed.saved_epochs == cm.saved_epochs
    assert resumed.monitor_mode == cm.monitor_mode
    assert resumed.keep_last_n == cm.keep_last_n

# MISSING_FILES.md — AgriVisionNet

Files that were **discussed, requested, or referenced in planning
documents but never actually implemented**. Listed here explicitly rather
than invented, per instruction. Cross-checked directly against the actual
filesystem at packaging time (not from memory) — see the verification
commands run immediately before this file was written.

None of these existing were required for the ZIP to be "complete" — the
ZIP contains every file that WAS implemented, verified against
`PROJECT_TREE.md` (62/62 match, zero discrepancies in either direction).
This file exists only so nothing is silently assumed to exist that doesn't.

---

## `training/`

| File | Status |
|---|---|
| `training/callbacks.py` | **Not implemented.** Planned: `NaNGuard` (loss-value NaN detection) and `GradientMonitor` (zero-gradient-per-parameter flagging), decision logic separated from tensor extraction, following the same pattern already used in `training/checkpoint_manager.py`. See `PROJECT_HANDOFF.md` Section 18 — this was the explicitly-named next task at handoff time. |
| `training/logger.py` | **Not implemented.** Planned: TensorBoard + CSV logging facade. |
| `training/config.py` | **Not implemented.** Planned: extract config-loading/resolution responsibility out of `training/train.py` into its own module. |
| `training/cli.py` | **Not implemented as a separate file.** The CLI interface exists today as `training/train.py` (`build_arg_parser`, `resolve_config`, `main`) — a rename to `cli.py` was requested but not carried out. **Do not assume both files exist; only `train.py` does.** |
| `training/scheduler.py` | **Not implemented as a separate file.** LR schedule logic exists today as `training/lr_schedule.py` (`cosine_warmup_multiplier`) — a rename to `scheduler.py` was requested but not carried out. |
| `training/checkpoint.py` | **Not implemented as a separate file.** Checkpoint bookkeeping exists today as `training/checkpoint_manager.py` (`CheckpointManager`) — a rename to `checkpoint.py` was requested but not carried out. |

## `evaluation/`

**Entire directory is empty** (contains only `.gitkeep`). None of the
following were implemented:
- `evaluation/metrics.py` — accuracy/precision/recall/F1/AUROC
- `evaluation/calibration.py` — ECE, reliability diagram, calibration curve
- `evaluation/confusion.py` — confusion matrix computation

## `frontend/`

**Entire directory is empty** (contains only `.gitkeep`). Phase 9
(React + Tailwind dashboard) has not started. No files exist.

## `backend/`

**Entire directory is empty** (contains only `.gitkeep`). Phase 8
(FastAPI backend) has not started. No files exist.

## `research/`

| File | Status |
|---|---|
| `research/threats_to_validity.md` | **Not implemented.** Requested (Internal/External/Construct/Conclusion validity sections with specific bullet points) but not started — see `PROJECT_HANDOFF.md` Section 6/10. |

## `tests/`

The 10-point "Training Verification Suite" requested alongside Phase 3's
expansion was **not implemented**. None of the following exist:
- `tests/test_training/test_single_batch_overfit.py`
- `tests/test_training/test_gradient_flow.py`
- `tests/test_training/test_checkpoint_resume.py` (full Trainer-level; note `test_checkpoint_manager.py` DOES exist and tests the bookkeeping layer only, which is not the same thing)
- `tests/test_training/test_scheduler_resume.py`
- `tests/test_training/test_loss_masking.py`
- `tests/test_training/test_mixed_precision.py`
- `tests/test_training/test_seed_reproducibility.py`
- `tests/test_training/test_nan_detection.py`
- `tests/test_training/test_zero_gradients.py`
- `tests/test_training/test_model_save_load.py`

What DOES exist in `tests/test_training/`: `test_checkpoint_manager.py`,
`test_early_stopping.py`, `test_lr_schedule.py` — all executed and passing,
covering the torch-free bookkeeping/scheduling logic only.

## Also empty, not requested in this packaging but worth knowing about

`explainability/`, `recommendation/`, `weights/`, `outputs/`, `data/` —
all contain only `.gitkeep`, no implementation. These correspond to Phases
5–7 and 10, not started. Not in your requested directory list, included
here only so nothing is assumed present by omission.

---

## What this means for continuing the project

Everything above is a **known, previously-documented gap** — none of it is
new information, all of it was already tracked in `PROJECT_HANDOFF.md`
Sections 6, 10, and 17 (pending tasks / TODO list). This file is a
narrower, purely file-existence-focused cross-check of that same
information, produced by actually inspecting the filesystem rather than
copying from the handoff doc's prose.

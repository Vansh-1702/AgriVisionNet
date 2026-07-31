# AgriVisionNet

**Task- and Hierarchy-Aware Evidential Fusion for Hybrid CNN–Swin
Multi-Task Crop Disease Analysis** *(frozen architecture)*

## What this is

An integrated-systems research contribution (not a claim of inventing
CNN-Transformer fusion or evidential deep learning — see
`research/design_rationale.md` for the full positioning against nearest
prior work: TMC/ETMC, MMTM, AFF/iAFF, Mobile-Former).

Two mechanisms constitute the actual, checkable delta:
1. **Task-aware fusion** — a separate reliability-conditioned gate per task
   head (crop / disease / severity), not one shared gate.
2. **Hierarchy-aware discounting** — severity's evidential output is
   Dempster-Shafer-discounted by the disease head's own uncertainty.

## Status

**Phase 1 — DONE (frozen, no further architecture changes without approval): core architecture**
- `configs/config.yaml` — single source of truth, fully documents citations/positioning inline
- `models/backbone.py` — EfficientNet-B0 + Swin-Tiny dual branch (standard, not novel)
- `models/fusion/attention_blocks.py` — SE + CBAM building blocks (standard, not novel)
- `models/fusion/task_hierarchy_fusion.py` — **the claimed contribution**: per-task evidential gating
- `models/evidential.py` — EDL Dirichlet head, DS discounting operator, KL annealing (Sensoy et al. 2018; Shafer 1976)
- `models/heads.py` — evidential multi-task heads + hierarchical discounting
- `models/agrivisionnet.py` — assembled end-to-end model
- `losses/evidential_loss.py` — type-II MLE + annealed KL loss per task, plus auxiliary reliability-head supervision
- `tests/test_evidential.py`, `tests/test_fusion.py` — pytest suite (requires `torch`+`timm`; see caveat below)
- `scripts/verify_core_math_numpy.py` — **executed in this sandbox, 18/18 checks pass** — validates the EDL/DS-discount math with plain numpy, since torch isn't installed here
- `research/design_rationale.md` — full literature positioning, ablation plan, explicit non-claims
- `research/reproducibility.md` — every field tagged MEASURED / PINNED / PENDING, no fabricated numbers
- `research/experiments.md` — baselines, metrics, the 7 required ablations (2 flagged as needing new code in Phase 4), statistical protocol

**Phase 2 — DONE: data pipeline (engineering only, architecture untouched)**
- `datasets/base_dataset.py` — unified `Sample`/`BaseCropDataset` interface; adding a dataset = implement one `_scan()` method
- `datasets/registry.py` — name-based dataset registry (`@register_dataset`)
- `datasets/plantvillage.py`, `paddy_doctor.py`, `rice_disease.py` — concrete loaders, each with **documented, explicitly-flagged directory-structure assumptions** (no real dataset was downloaded/inspected — see `datasets/README.md`)
- `datasets/label_space.py` — cross-dataset label unification (e.g. "rice" from two different datasets resolves to the same integer)
- `datasets/integrity.py` — corrupt/zero-byte/duplicate/missing-file detection, class-imbalance ratio
- `datasets/statistics.py` — per-crop/disease/source counts, severity coverage, image-size distribution
- `datasets/splitting.py` — seeded stratified train/val/test split, persisted to disk, rare-class handling with an explicit warning
- `datasets/transforms.py` — Albumentations pipeline builder (unverified — `albumentations` not installed in this sandbox) + torch-free MixUp/CutMix sampling math (verified)
- `datasets/unified.py` — `UnifiedCropDataset`, the actual training-facing object combining all enabled datasets
- `tests/test_datasets/` — full pytest suite against synthetic (non-experimental) fixture directories
- **Known open item, flagged not hidden:** none of the three documented datasets provide severity labels. **Resolved decision:** manual annotation only (`research/severity_annotation_protocol.md`), with pipeline support now built (`datasets/severity_annotations.py`, `UnifiedCropDataset(..., severity_annotations_csv=...)`) — but no annotation has actually been performed yet. A real bug (overlay being silently bypassed by `__getitem__`) was caught and fixed while wiring this in.

**Phase 3 — NOT YET BUILT:** training framework (`train.py`, AMP, checkpointing, early stopping, TensorBoard/W&B)

**Phase 4 — NOT YET BUILT:** baseline comparison (ResNet50 / EfficientNet-only / Swin-only / this hybrid) + the 7 ablations listed in `research/experiments.md` (2 of which need new code before they're runnable — see that doc)

**Phase 5 — NOT YET BUILT:** unknown-disease detection (the evidential uncertainty already computed here is the natural signal — likely supersedes the originally-planned separate Mahalanobis module; needs a decision when we get there)

**Phase 6 — NOT YET BUILT:** explainability (Grad-CAM + attention rollout, weighted by branch reliability)

**Phase 7 — NOT YET BUILT:** treatment recommendation engine (JSON-backed, RAG/LLM-ready), confidence-calibrated by the same uncertainty signal

**Phase 8 — NOT YET BUILT:** FastAPI backend

**Phase 9 — NOT YET BUILT:** React + Tailwind frontend

**Phase 10 — NOT YET BUILT:** automated research output generation (confusion matrices, ROC/PR curves, ablation tables, Grad-CAM figures)

**Phase 11 — NOT YET BUILT:** IEEE paper draft

## Known sandbox limitation

This authoring environment has no GPU/network access. Two consequences,
kept separate since they cover different code:
- **Phase 1 (model):** `torch`/`timm` not installed → `tests/test_evidential.py`,
  `tests/test_fusion.py`, and `models/agrivisionnet.py`'s shape-check are
  **unexecuted**. `scripts/verify_core_math_numpy.py` (18/18 checks) validates
  the underlying EDL/DS-discount math only, not the `nn.Module` wiring.
- **Phase 2 (data):** most of the pipeline (scanning, label unification,
  integrity, statistics, splitting, MixUp/CutMix math) uses no torch and
  **was actually executed** against synthetic fixture directories in this
  sandbox — see `datasets/README.md`'s verification section for exactly
  what was and wasn't checked. `albumentations` specifically is not
  installed, so `datasets/transforms.py`'s `build_train_transform`/
  `build_eval_transform` are unverified.

Run `pip install -r requirements.txt && pytest tests/` locally — covering
both `tests/test_evidential.py`/`test_fusion.py` (Phase 1) and
`tests/test_datasets/` (Phase 2) — before starting Phase 3.

## Next step

Phase 3 (training framework) is next. Two open items block or shape it:
1. **No real dataset has been downloaded yet** — `datasets/README.md`
   documents assumed (unverified) directory structures for PlantVillage,
   Paddy Doctor, and "Rice Disease" (the latter name is genuinely
   ambiguous between two public datasets — see that doc). Confirm which
   you actually have before Phase 3 runs against real data instead of
   fixtures.
2. **No dataset provides severity ground truth** — the severity head has
   nothing to train on until this is resolved (three options laid out in
   `datasets/README.md`'s data-gap section). This affects Phase 3/4's
   scope materially and should be decided before training starts, not
   discovered partway through it.

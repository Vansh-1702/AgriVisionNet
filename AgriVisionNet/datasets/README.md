# Data Pipeline (Phase 2) — Assumptions, Gaps, and Usage

**No real dataset has been downloaded or inspected in this project.**
Everything below is built against **documented, publicly-known directory
structures**, explicitly flagged as unverified where relevant. This is
exactly what was asked for when real datasets weren't available — treat
every "unverified" tag below as a required check before trusting a
training run, not as a formality.

## Per-dataset assumptions (see each module's docstring for full detail)

| Dataset | Loader | Assumed layout | Confidence |
|---|---|---|---|
| PlantVillage | `plantvillage.py` | `<root>/{Crop}___{Disease}/*.jpg` (triple underscore) | Moderate-high — this is the standard convention across the original release and common Kaggle mirrors, but not checked against an actual download in this project |
| Paddy Doctor | `paddy_doctor.py` | `<root>/train_images/<label>/*.jpg` + optional `train.csv` | Moderate — matches the Kaggle competition's documented structure, not verified |
| "Rice Disease" | `rice_disease.py` | **Genuinely ambiguous** — at least two differently-structured public datasets share this name. Two layouts supported (split-subfolder and flat-subfolder); loader tries both | **Low** — this is the one loader most likely to need adjustment once you confirm which specific source you mean |

## Severity data gap — resolution in progress

**Decision frozen:** severity ground truth will come from **manual
annotation only** — never fabricated, derived, or heuristically generated
as primary supervision. Full protocol: `research/severity_annotation_protocol.md`.

**No annotation has been performed yet.** What exists now is the pipeline
support for it: `datasets/severity_annotations.py` loads the annotation CSV
(schema defined in the protocol doc) and joins `resolved_severity` onto
`Sample` objects via `UnifiedCropDataset(..., severity_annotations_csv=...)`,
reusing the existing `severity_name=None → MISSING_LABEL(-1)` mechanism
from Phase 2 — no new masking logic, no architecture change. Verified in
this sandbox against a clearly-labeled schema-test fixture (never real
annotation data): correct joining, correct handling of unresolved/blank
rows, and a crop/disease mismatch safety check that refuses to attach a
label if the CSV's row doesn't actually agree with the `Sample` it matched
by filepath (catches a wrong-join scenario rather than silently trusting it).

A real, pre-existing bug was caught and fixed while wiring this in:
`UnifiedCropDataset.__getitem__`/`get_raw_sample` previously read samples
from each sub-dataset's original `.samples` list, bypassing any overlay
applied to the flattened `_all_samples` list — meaning a severity overlay
would have been silently ignored at the exact point it mattered. Both
methods now read from `_all_samples` consistently.

A lesion-area proxy, if built later, is explicitly an **auxiliary
experiment only** — never presented as equivalent to the manually-annotated
set (see protocol doc, Section 9).

## Why one unified splitting/statistics/integrity layer, not per-dataset

Every loader returns `Sample` objects through the exact same interface
(`datasets/base_dataset.py`), so `datasets/splitting.py`,
`datasets/statistics.py`, and `datasets/integrity.py` are written once and
work identically across all three (and any future) datasets — the whole
point of Phase 2 objective 1 (unified interface) and objective 8 (add new
datasets with minimal code changes).

## Why train/val/test splits are re-computed, not reused from source

Paddy Doctor ships its own train/test split (and the ambiguous "Rice
Disease" candidates may too); PlantVillage's common Kaggle mirrors
sometimes ship a pre-augmented "train/valid" split. **This project
deliberately ignores those and re-splits everything from scratch** via
`datasets/splitting.py`, for two reasons: (1) a fair 4-way baseline
comparison (Section 1 of `research/experiments.md`) requires every model
to see literally the same held-out samples, which isn't guaranteed if each
dataset's own split logic differs; (2) some "pre-split" releases (e.g. the
common "New Plant Diseases Dataset (Augmented)" mirror) apply augmentation
*before* splitting, which can leak augmented near-duplicates of a training
image into the validation set — re-splitting from the original samples
avoids inheriting that risk sight-unseen.

## What is actually verified vs. not (Phase 2)

Real execution happened in this authoring sandbox for every pure-Python
piece of this pipeline (no torch/GPU required): dataset scanning against
synthetic fixture directories, label-space cross-dataset unification,
integrity checks (corrupt/zero-byte/duplicate/missing-file detection, each
against a deliberately-broken fixture), dataset statistics, stratified
splitting (including the rare-class-warning path and determinism), and the
MixUp/CutMix sampling math. **Not verified**: `BaseCropDataset.__getitem__`'s
tensor output path (requires torch, not installed here) and
`transforms.py`'s `build_train_transform`/`build_eval_transform` (require
`albumentations`, not installed here). Run `pytest tests/test_datasets/`
locally, with the full `requirements.txt` installed, before Phase 3.

## Adding a new dataset (Phase 2 objective 8, concretely)

1. Write `datasets/your_dataset.py` with a class extending `BaseCropDataset`,
   decorated `@register_dataset("your_name")`, implementing only `_scan()`.
2. Add one import line to `datasets/__init__.py`.
3. Add a block to `configs/config.yaml`'s `data.datasets` list.
4. Nothing else changes — `UnifiedCropDataset`, splitting, statistics, and
   integrity checks all work automatically once step 1–3 are done, because
   they operate on the shared `Sample`/`BaseCropDataset` interface, not on
   any dataset-specific logic.

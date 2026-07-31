# Reproducibility Checklist

**How to read this document:** every field is tagged.
- **[MEASURED]** — actually run and observed in this project; a command is shown.
- **[PINNED]** — a specific version/setting you should lock in now, chosen with a stated reason, but not yet empirically verified end-to-end (this authoring sandbox has no GPU/network — see `README.md`).
- **[PENDING Phase N]** — cannot be honestly filled in yet. Do not replace with an invented number. Fill in only when that phase actually produces the number, and cite the run (log path, git commit, date).

This distinction exists because a reproducibility document that quietly mixes guessed numbers with real ones is worse than no document — it gives false confidence to whoever tries to reproduce the paper.

---

## 1. Software environment

| Component | Version | Status |
|---|---|---|
| Python | 3.12.3 | **[MEASURED]** in this authoring sandbox (`python3 --version`). **[PINNED for training]**: use 3.11.x on the actual training machine instead — PyTorch 2.1.0's officially tested range is 3.8–3.11; 3.12 support was still stabilizing at that PyTorch version. Re-verify against the PyTorch release notes for whichever version you actually install. |
| PyTorch | 2.1.0 (`+cu121` build) | **[PINNED, not yet installed here]** — chosen for stable `nn.MultiheadAttention` batch_first support and mature AMP; not verified against this repo's actual forward pass in this sandbox (no GPU/network). |
| torchvision | 0.16.0 | **[PINNED]** — must match the PyTorch 2.1.0 minor version exactly (torchvision versioning is coupled to torch). |
| timm | 0.9.12 | **[PINNED]** — **critical pin**, not a suggestion: `models/backbone.py:SwinBranch.forward` defensively handles NCHW / NHWC / token-sequence outputs specifically because timm's Swin `features_only=True` output shape has changed across releases. Confirm this exact version's output shape (see Step 3 of Section 7 below) before trusting any downstream shape assumption. |
| transformers | 4.38.0 | **[PINNED]** — only used if a HuggingFace-hosted checkpoint is substituted for a timm one; not on the critical path for Phase 1–3 as currently scoped. |
| albumentations | 1.4.0 | **[PINNED]** — Phase 2 dependency, not yet used. |
| CUDA | 12.1 | **[PINNED]** — matches the `+cu121` wheel above. If your GPU/driver only supports CUDA 11.8, switch the entire PyTorch/torchvision pin to the `+cu118` build instead; do not mix CUDA minor versions across the stack. |
| cuDNN | bundled with the `+cu121` PyTorch wheel (≈8.9.x) | **[PINNED]** — do not install a separate system cuDNN; the pip wheel vendors its own. |
| OS | Ubuntu 24.04.4 LTS | **[MEASURED]** in this authoring sandbox (`/etc/os-release`). Reasonable to standardize the training machine on the same LTS release. |
| GPU / hardware used for training | — | **[PENDING Phase 3]** — no training has occurred yet. Record exact GPU model, VRAM, driver version, and node count here once Phase 3 runs. |
| Expected training time per model | — | **[PENDING Phase 3]** — do not estimate this without at least one measured epoch on real hardware; a guessed number here is exactly the kind of false-confidence entry this document exists to prevent. Once Phase 3 exists, record wall-clock time per epoch and total epochs to convergence for each of the 4 models in the comparison table (`research/experiments.md`). |

### Exact pinned `requirements.txt` vs. this table
`requirements.txt` currently uses `>=` minimum-version constraints (chosen deliberately in Phase 1 for author-side flexibility while no GPU was available to test against). **Before Phase 3 training begins, freeze a `requirements-lock.txt`** with exact `==` pins matching this table, generated via `pip freeze` on the actual training machine, and commit it alongside the trained checkpoints it produced. A reproducibility claim tied to `>=` ranges is not a real reproducibility claim.

---

## 2. Random seed strategy

**[PINNED, implemented in `configs/config.yaml`: `project.seed: 42`]**

A single global seed must be propagated to every source of randomness:

```python
import random, os
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Additional seed-sensitive points that are easy to silently miss, and must be
handled explicitly in Phase 2/3 (**[PENDING implementation]**, listed here so
it isn't forgotten):
- `DataLoader(..., worker_init_fn=..., generator=torch.Generator().manual_seed(seed))` — otherwise multi-worker data loading reintroduces nondeterminism even with the global seed set.
- Albumentations' own RNG — must be seeded separately (`random.seed`/`np.random.seed` cover it, since Albumentations uses those, but confirm this against the installed version's source).
- MixUp/CutMix's beta-distribution sampling — confirm it draws from the already-seeded `np.random`/`torch` RNG, not an unseeded one, when implemented in Phase 2.
- Any `torch.utils.data.random_split` or shuffling used for train/val splits — must use the seeded generator, and the resulting split indices should be saved to disk (see Section 6) so the exact split is reproducible even if the splitting *code* changes later.

**Multi-run protocol** (see `research/experiments.md`): every reported number is a mean ± std over 5 runs at seeds `{42, 43, 44, 45, 46}`, not 5 arbitrary reruns — fixing the seed *set* in advance prevents seed-shopping (running until a favorable seed appears).

## 3. Deterministic settings

**[PINNED, not yet verified end-to-end without a GPU]**

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
# some cuBLAS ops require this env var set BEFORE the CUDA context is created:
# CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Known trade-off to document, not hide: `cudnn.deterministic=True` typically
costs 10–30% throughput versus `cudnn.benchmark=True` on convolution-heavy
backbones like EfficientNet. Decide per-run whether a given result is a
"deterministic reproducibility run" or a "throughput run," and label it as
such in logs — don't conflate the two.

`warn_only=True` is used because `nn.MultiheadAttention`'s backward pass (used in earlier, now-superseded fusion attempts) and some Swin ops may not have a fully deterministic CUDA kernel in every PyTorch version; a hard `warn_only=False` could crash training on an op with no deterministic implementation. Record any warning raised here — a warning here means true bit-exact reproducibility across GPUs is not guaranteed for that op, only same-seed-same-hardware reproducibility.

## 4. Dataset versions

**[PENDING Phase 2 — dataset selection not yet finalized with the user]**

Do **not** fill in specific download links/hashes here speculatively. When Phase 2 starts, this section must record, per dataset actually used:
- Exact source (Kaggle dataset slug + version number, or official repo commit hash)
- Download date
- SHA-256 of the downloaded archive
- Exact class list and per-class image counts as actually loaded (auto-generated by the dataset-statistics script Phase 2 must produce — see `research/experiments.md` Section on dataset stats)
- Train/val/test split strategy and the exact saved split-index file (Section 6)

This project's brief names PlantVillage, Rice Disease, Paddy Doctor, Tomato, Apple, and Corn datasets; **which of these you actually have access to was asked and not yet answered** — Phase 2 cannot honestly proceed past scaffolding without this.

## 5. Hardware used

**[PENDING Phase 3]** — see the table in Section 1. Record here once real training happens: GPU model(s), VRAM, CPU, RAM, storage type (affects data-loading throughput measurably for large image datasets), and whether training was single-GPU or distributed.

## 6. Folder structure (current, Phase 1 state)

```
AgriVisionNet/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── models/
│   ├── backbone.py
│   ├── evidential.py
│   ├── heads.py
│   ├── agrivisionnet.py
│   └── fusion/
│       ├── attention_blocks.py
│       └── task_hierarchy_fusion.py
├── losses/
│   └── evidential_loss.py
├── datasets/                    (Phase 2 — done)
│   ├── README.md                # assumptions, known gaps, verification status
│   ├── base_dataset.py          # unified Sample/BaseCropDataset interface
│   ├── registry.py
│   ├── label_space.py           # cross-dataset label unification
│   ├── plantvillage.py
│   ├── paddy_doctor.py
│   ├── rice_disease.py
│   ├── unified.py               # UnifiedCropDataset -- the training-facing object
│   ├── integrity.py
│   ├── statistics.py
│   ├── splitting.py
│   └── transforms.py
├── scripts/
│   └── verify_core_math_numpy.py
├── tests/
│   ├── test_evidential.py       # Phase 1 (requires torch+timm, unexecuted here)
│   ├── test_fusion.py           # Phase 1 (requires torch+timm, unexecuted here)
│   └── test_datasets/           # Phase 2 (torch-free logic executed here; see datasets/README.md)
│       ├── conftest.py
│       ├── test_base_dataset.py
│       ├── test_plantvillage.py
│       ├── test_paddy_doctor.py
│       ├── test_rice_disease.py
│       ├── test_label_space.py
│       ├── test_integrity.py
│       ├── test_statistics.py
│       ├── test_splitting.py
│       ├── test_transforms.py
│       └── test_unified.py
├── research/
│   ├── design_rationale.md
│   ├── reproducibility.md   (this file)
│   └── experiments.md
├── data/            (empty — real dataset downloads go here; see datasets/README.md for expected structure per dataset)
├── training/        (empty — Phase 3)
├── evaluation/      (empty — Phase 3/4)
├── explainability/  (empty — Phase 6)
├── recommendation/  (empty — Phase 7)
├── backend/         (empty — Phase 8)
├── frontend/        (empty — Phase 9)
├── weights/         (empty — will hold checkpoints from Phase 3 on; NOT committed to git, document the retrieval path instead — see Section 8)
└── outputs/         (empty — will hold generated figures/tables from Phase 10)
```
As Phase 3+ add real content to the still-empty directories above, this section must be updated in the same commit — a stale folder-structure listing is exactly the kind of drift this checklist exists to catch (as happened with `README.md` and the missing `design_rationale.md` in Phase 1, before that review cycle).

## 7. How to reproduce Phase 1 (what actually exists today)

```bash
# 1. Clone / unpack the repository, then:
cd AgriVisionNet
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --break-system-packages   # or omit the flag in a venv

# 2. Run the torch-free math verification (works even without GPU):
python3 scripts/verify_core_math_numpy.py
# Expected: "SUMMARY: 18/18 checks passed" -- this is the ONLY command in
# this section that has actually been executed and confirmed in this
# project so far.

# 3. Run the full pytest suite (requires the torch/timm install from step 1;
#    NOT yet executed in this project -- do this first before Phase 2):
pytest tests/ -v

# 4. Shape sanity-check the assembled model directly:
python3 -m models.agrivisionnet
```

## 8. How to reproduce every experiment / table / figure

**[PENDING Phases 3, 4, 10]** — the scripts referenced below do not exist yet. This section is written now as the **binding interface specification** those phases must implement, so "reproduce the paper" has one consistent command surface from day one rather than being retrofitted later.

```bash
# Training (Phase 3) -- one call per model in the comparison table
python training/train.py --config configs/config.yaml --model resnet50   --seed 42 --run-name resnet50_s42
python training/train.py --config configs/config.yaml --model efficientnet_b0 --seed 42 --run-name effb0_s42
python training/train.py --config configs/config.yaml --model swin_tiny  --seed 42 --run-name swint_s42
python training/train.py --config configs/config.yaml --model agrivisionnet --seed 42 --run-name ours_s42
# ... repeated for seeds {42,43,44,45,46} per research/experiments.md's
# statistical-validation protocol -- 5 seeds x 4 primary models = 20 runs
# minimum, before any ablation.

# Ablations (Phase 4) -- one call per row of research/experiments.md's A1-A7
python training/train.py --config configs/config.yaml --model agrivisionnet \
    --ablation shared_fusion --seed 42 --run-name ablation_A1_s42
# (see research/experiments.md for the full A1-A7 flag mapping)

# Evaluation + metrics table (Phase 4)
python evaluation/evaluate.py --run-name ours_s42 --output outputs/tables/main_results.csv

# Figures (Phase 10) -- each figure has its own explicit command, not one
# monolithic "make figures" script, so a single figure can be regenerated
# without rerunning everything else:
python evaluation/plot_roc_pr.py          --run-name ours_s42 --out outputs/figures/roc_pr.png
python evaluation/plot_confusion_matrix.py --run-name ours_s42 --out outputs/figures/confusion.png
python evaluation/plot_calibration.py      --run-name ours_s42 --out outputs/figures/calibration.png
python evaluation/plot_training_curves.py  --run-name ours_s42 --out outputs/figures/training_curves.png
python explainability/gradcam_report.py    --run-name ours_s42 --out outputs/figures/gradcam/
python evaluation/plot_ablation_chart.py   --results outputs/tables/ablation_results.csv --out outputs/figures/ablation_chart.png
```

Every run must write its exact resolved config (after any `--ablation`/CLI overrides are applied) to `outputs/runs/<run-name>/resolved_config.yaml`, so a figure or table can always be traced back to the exact config that produced it — this is a Phase 3 requirement, recorded here now so it isn't skipped when that phase is built.

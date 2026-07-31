# PROJECT_HANDOFF.md — AgriVisionNet

**Written:** end of an extended design+build session, mid-Phase-3.
**Purpose:** let a new engineer (or a new Claude conversation with zero prior
context) continue this project with no information loss.

---

## 1. Project Overview

**AgriVisionNet** is a final-year B.Tech research project, also intended for
submission to a Scopus-indexed IEEE conference. It is an intelligent
agricultural decision-support system for crop disease diagnosis — not a
simple classifier. The system is meant to eventually perform: crop
identification, disease classification, severity estimation, explainable AI,
unknown-disease detection, treatment recommendation, and serve all of this
through a web dashboard. This handoff covers the **model architecture and
data/training engineering**, not the frontend/backend/paper phases, which
have not started.

## 2. Goals

- Build something **genuinely publishable** — technically sound,
  reproducible, defensible under peer review — not just a working demo.
- Explicit priority order set by the project owner: **scientific honesty >
  novelty-maximizing > implementation speed.** Multiple rounds of this
  project involved *rejecting* an initially-built architecture because
  self-critique found it wasn't sufficiently novel (see Section 12).
- Every module must be literature-grounded, every claim must be either
  executed-and-verified or explicitly marked unverified, and no fabricated
  benchmarks/results/passing tests are ever acceptable.

## 3. Frozen Architecture

**Name:** *Task- and Hierarchy-Aware Evidential Fusion for Hybrid CNN–Swin
Multi-Task Crop Disease Analysis*

**Status: FROZEN.** Do not modify without explicit owner approval (see
Section 19). This was arrived at only after multiple rounds of adversarial
self-review (acting as a hostile IEEE reviewer) rejected earlier, more
"novel-sounding" designs as too close to prior work.

**What it is, precisely:**
1. Dual backbone: **EfficientNet-B0** (local/texture features) +
   **Swin-Tiny** (global/context features). Standard, not claimed novel.
2. Each branch gets a **lightweight auxiliary Dirichlet-evidence head**
   (Evidential Deep Learning, Sensoy et al. NeurIPS 2018) trained on the
   disease label, producing a per-branch **uncertainty mass** `u = K/S`
   (K=num classes, S=Dirichlet strength).
3. **Task-aware fusion**: a *separate* reliability-conditioned gate per task
   head (crop/disease/severity) — NOT one shared gate — mixes the two
   branches' pooled features, weighted by `(1 - uncertainty)` per branch.
   This is the first of two claimed (modest, incremental) contributions.
4. Each of the three task heads (crop, disease, severity) is itself an
   **evidential head** (Dirichlet output), not a plain softmax.
5. **Hierarchical discounting** (second claimed contribution): the
   severity head's Dirichlet output is **Dempster-Shafer discounted**
   (Shafer, 1976; subjective-logic discounting operator) by the disease
   head's own uncertainty — formalizing "don't trust severity if disease
   identity itself is uncertain."

**Honest novelty framing (do not deviate from this in any paper draft):**
this is an **integrated systems contribution**, not an architectural
breakthrough. No individual mechanism (EfficientNet, Swin, EDL, DS theory,
channel/spatial attention, gated fusion) is claimed as new. Nearest prior
work is TMC/ETMC (Han et al., ICLR 2021), MMTM (Joze et al., CVPR 2020),
AFF/iAFF (Dai et al., WACV 2021), Mobile-Former (Chen et al., CVPR 2022).
Full comparison table and delta argument: `research/design_rationale.md`.

**Rejected/superseded designs** (for context, do not resurrect without
re-deriving why they were rejected): a "custom Adaptive Feature Fusion
Module" combining SE+CBAM+bidirectional-cross-attention+dynamic-gating was
built first, then **scrapped** after literature review showed it was
essentially a recombination of AFF/MMTM/Mobile-Former with no real delta.

## 4. Folder Structure (exact, verified via `find` at handoff time)

```
AgriVisionNet/
├── README.md                              # top-level status tracker -- KEEP UPDATED
├── configs/config.yaml                    # single source of truth for all hyperparameters/paths
├── models/
│   ├── backbone.py                        # EfficientNet-B0 + Swin-Tiny dual branch
│   ├── evidential.py                      # EDL Dirichlet head, DS discount op, KL annealing
│   ├── heads.py                           # MultiTaskEvidentialHeads
│   ├── agrivisionnet.py                   # AgriVisionNet (assembled model), build_model()
│   └── fusion/
│       ├── attention_blocks.py            # ChannelAttention (SE), SpatialAttention (CBAM) -- standard, not novel
│       └── task_hierarchy_fusion.py       # PerTaskReliabilityGate, TaskHierarchyEvidentialFusion -- THE claimed contribution
├── losses/evidential_loss.py              # MultiTaskEvidentialLoss, build_loss()
├── datasets/                              # Phase 2 -- see Section 15
│   ├── README.md                          # assumptions/gaps/verification-status doc
│   ├── base_dataset.py, registry.py, label_space.py
│   ├── plantvillage.py, paddy_doctor.py, rice_disease.py
│   ├── unified.py, integrity.py, statistics.py, splitting.py, transforms.py
│   └── severity_annotations.py            # manual-annotation CSV loader/overlay/kappa
├── training/                              # Phase 3 -- IN PROGRESS, see Section 6
│   ├── seeding.py, early_stopping.py, lr_schedule.py, checkpoint_manager.py
│   ├── trainer.py, train.py
│   └── (NOT YET BUILT: callbacks.py, logger.py, config.py, cli.py rename -- see Section 10)
├── evaluation/                            # directory exists, EMPTY -- Phase 3 evaluation metrics not built yet
├── scripts/verify_core_math_numpy.py      # torch-free math verification, actually executed, 18/18 pass
├── research/
│   ├── design_rationale.md                # literature positioning, delta table, non-claims
│   ├── reproducibility.md                 # MEASURED/PINNED/PENDING-tagged repro checklist
│   ├── experiments.md                     # baselines, metrics, A1-A7 ablations, stats protocol
│   └── severity_annotation_protocol.md    # manual annotation scale/guidelines/CSV schema
│   └── (NOT YET BUILT: threats_to_validity.md -- requested, not started)
├── tests/
│   ├── test_evidential.py, test_fusion.py           # Phase 1, torch-dependent, UNVERIFIED (no torch in sandbox)
│   ├── test_datasets/                                # Phase 2, torch-free logic, EXECUTED & PASSING
│   └── test_training/                                # Phase 3, PARTIAL -- see Section 6
├── data/                                  # empty -- no real dataset downloaded, ever, in this project
├── explainability/, recommendation/, backend/, frontend/  # empty -- Phases 6-9, not started
└── weights/, outputs/                     # empty
```

## 5. Every Completed Phase

- **Phase 1 (Model Architecture): FROZEN, approved.** Built, math-verified
  via numpy (18/18 checks in `scripts/verify_core_math_numpy.py`), but the
  actual `nn.Module` forward pass has **never been run** — no torch in any
  sandbox used for this project so far.
- **Phase 2 (Data Pipeline): approved.** Built and **actually executed**
  against synthetic fixture directories (never real downloaded data — see
  Section 15). All logic that doesn't need torch/albumentations was run and
  passed, including catching and fixing real bugs (Section 9).
- **Severity annotation protocol: approved, pipeline support built and
  tested.** See Section 16. No real annotation has been performed yet —
  this is a protocol + code readiness deliverable, not actual data.

## 6. Current Phase and Exact Progress

**Phase 3 (Training Framework): IN PROGRESS, NOT COMPLETE.** This is the
most important section for continuation — read carefully.

**What exists and is TESTED (executed, passing) as of handoff:**
- `training/seeding.py` — `set_seed()`, `seed_worker()` (torch import
  guarded; numpy/random parts work without torch)
- `training/early_stopping.py` — `EarlyStopping` class, fully tested
  (mode=min/max, min_delta noise resistance, state_dict resume round-trip)
- `training/lr_schedule.py` — `cosine_warmup_multiplier()`, fully tested
  (warmup monotonicity, cosine decay monotonicity, clamping, invalid configs)
- `training/checkpoint_manager.py` — `CheckpointManager` class, fully
  tested (best-tracking, pruning, resolved_config.yaml traceability,
  state_dict resume)
- `training/train.py` — CLI (`build_arg_parser`, `resolve_config`, `main`).
  **`resolve_config()` is torch-free and WAS tested** (seed override,
  ablation override application, NotImplementedError for A1/A5, missing-file
  handling, argparse validation — all passing). `main()` itself has never
  run (needs torch).
- `training/trainer.py` — `Trainer` class. **Written completely, NEVER
  EXECUTED** (no torch). Two real bugs were caught by manually
  cross-checking against the actual pre-existing `losses/evidential_loss.py`
  and `models/agrivisionnet.py` signatures (see Section 9) — fixed before
  they could cause a runtime failure on first real run.

**What the project owner asked for in the most recent message (a
restructuring + expansion) that has NOT been started yet:**
- Rename `lr_schedule.py` → `scheduler.py`, `checkpoint_manager.py` →
  `checkpoint.py`, `train.py` → `cli.py` (requested naming convention)
- New: `training/callbacks.py` (NaN detection, gradient-flow monitoring —
  plan was to separate pure-Python decision logic from torch tensor
  extraction, same pattern as `checkpoint_manager.py`, so the decision
  logic is testable without torch)
- New: `training/logger.py` (TensorBoard + CSV logging facade)
- New: `training/config.py` (extract config-loading responsibility out of
  the CLI file)
- New: `evaluation/metrics.py`, `evaluation/calibration.py`,
  `evaluation/confusion.py` (accuracy/precision/recall/F1/AUROC via
  sklearn — available in sandbox; ECE/reliability diagram/calibration curve
  — pure numpy, all plannable to actually test here)
- New: `research/threats_to_validity.md` (Internal/External/Construct/
  Conclusion validity sections, specific bullet points given by owner)
- New/expanded test suite matching the 10-point "Training Verification
  Suite" the owner specified: single-batch-overfit, gradient-flow,
  checkpoint-resume, scheduler-resume, loss-masking, mixed-precision, seed-
  reproducibility, NaN-detection, zero-gradient-detection, save/load
  consistency. **Plan (not yet executed):** several of these are fully
  torch-dependent and must be written-but-unverified; a few (scheduler-
  resume math equivalence, seed-reproducibility of numpy/random state,
  NaN-detection decision logic, zero-gradient flagging logic given
  precomputed norms) are genuinely testable without torch and should
  actually be executed, following the exact separation pattern already
  used successfully in `checkpoint_manager.py`.

**I had just finished inspecting the existing file tree (per the "inspect
before extending" rule) and had not yet written any of the above when this
handoff was requested.** This is the exact resume point.

## 7. Every File Created (with purpose)

See Section 4's tree for the full list; key purposes not obvious from
filename alone:

| File | Purpose |
|---|---|
| `models/fusion/task_hierarchy_fusion.py` | **The actual paper contribution.** `PerTaskReliabilityGate` (one gate per task, not shared), `TaskHierarchyEvidentialFusion` (assembles branch evidence + gating). |
| `models/evidential.py` | `DirichletOutput` (dataclass), `EvidentialHead` (nn.Module producing Dirichlet params), `ds_discount()` (the Dempster-Shafer discounting operator — core of the hierarchy mechanism), `kl_annealing_coefficient()`. |
| `losses/evidential_loss.py` | `evidential_classification_loss()` (type-II MLE + annealed KL, per-task), `MultiTaskEvidentialLoss` (combines all 3 tasks + auxiliary branch-reliability supervision), `build_loss(cfg)` factory. |
| `datasets/base_dataset.py` | `Sample` (frozen dataclass, the universal record type), `BaseCropDataset` (torch-optional base class — this is why Phase 2 logic is testable without torch). `MISSING_LABEL = -1` sentinel defined here, matches `losses/evidential_loss.py`'s `ignore_index=-1`. |
| `datasets/label_space.py` | `normalize_label()` (handles cross-dataset naming collisions, e.g. "Corn_(maize)" vs "corn maize"), `LabelSpace` (bidirectional string↔int, deterministic fit, save/load). |
| `datasets/unified.py` | `UnifiedCropDataset` — the actual training-facing object. Combines all enabled datasets, fits/reuses label spaces, optionally overlays severity annotations. **A real bug was fixed here** (Section 9). |
| `datasets/severity_annotations.py` | `load_severity_annotations()`, `attach_severity_labels()` (returns NEW Sample objects, since Sample is frozen), `compute_interrater_kappa()` (quadratic-weighted Cohen's κ). |
| `training/checkpoint_manager.py` | `CheckpointManager` — pure-Python bookkeeping (which checkpoint is best, which to prune) deliberately separated from actual `torch.save`/`torch.load`, so it's testable without torch. This separation pattern should be REUSED for `callbacks.py`. |
| `scripts/verify_core_math_numpy.py` | Standalone numpy reimplementation of the EDL/DS-discount math, actually executed (18/18 pass) since torch isn't available. Not a substitute for real `nn.Module` tests. |

## 8. Important APIs/Classes/Functions (exact signatures, verified via grep at handoff time)

```python
# models/agrivisionnet.py
class AgriVisionNet(nn.Module):
    def __init__(self, cfg: dict)
    def forward(self, x: torch.Tensor, discount_severity: bool = True, return_aux: bool = False)
def build_model(cfg: dict) -> AgriVisionNet

# models/heads.py
class MultiTaskEvidentialHeads(nn.Module):
    def forward(self, fused: dict, discount_severity: bool = True) -> dict

# models/evidential.py
class DirichletOutput            # dataclass; has .uncertainty attribute (used as out["crop_dirichlet"].uncertainty)
class EvidentialHead(nn.Module)
def ds_discount(belief: torch.Tensor, uncertainty: torch.Tensor, ...)
def kl_annealing_coefficient(epoch: int, annealing_steps: int) -> float

# losses/evidential_loss.py
def evidential_classification_loss(...)
class MultiTaskEvidentialLoss(nn.Module):
    def forward(self, outputs: dict, targets: dict, epoch: int) -> dict   # NOTE: epoch is REQUIRED, positional-or-keyword
def build_loss(cfg: dict) -> nn.Module

# datasets/base_dataset.py
@dataclass(frozen=True)
class Sample:
    filepath: Path; crop_name: str; disease_name: str; source_dataset: str; severity_name: Optional[str] = None
class BaseCropDataset(_TorchDataset):   # subclass contract: implement _scan() -> List[Sample] ONLY
MISSING_LABEL = -1

# datasets/unified.py
class UnifiedCropDataset(_TorchDataset):
    def __init__(self, dataset_configs: List[dict], transform=None,
                 label_spaces: Optional[Dict[str, LabelSpace]] = None,
                 severity_annotations_csv: Optional[str] = None)
    def label_spaces(self) -> Dict[str, LabelSpace]
    def all_samples(self) -> List[Sample]
    def get_raw_sample(self, idx: int) -> Sample

# datasets/registry.py
def register_dataset(name: str)          # class decorator
def build_dataset(name: str, root: str, transform=None) -> BaseCropDataset

# datasets/splitting.py
def stratified_split(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                      seed=42, min_class_count_for_stratification=3) -> DatasetSplit

# training/checkpoint_manager.py
class CheckpointManager:
    def decide(self, epoch: int, monitored_value: float) -> dict   # PURE, no filesystem side effects
    def save_resolved_config(self, resolved_config: dict, writer=None) -> Path

# training/early_stopping.py
class EarlyStopping:
    def step(self, value: float, epoch: int) -> bool   # returns True if should stop NOW

# training/lr_schedule.py
def cosine_warmup_multiplier(epoch: int, total_epochs: int, warmup_epochs: int, min_lr_ratio: float = 0.0) -> float

# training/train.py
def resolve_config(args: argparse.Namespace) -> dict    # torch-free, tested
```

## 9. Bugs Found and Fixed (every one, with origin classified)

1. **Test-authoring bug (mine), not a code bug** — `scripts/verify_core_math_numpy.py`'s
   "concentrated evidence" check initially used evidence magnitude too small
   for its own asserted threshold (`u < 1e-3` needs `S > 6000` for `K=6`, but
   evidence was only ~500). Fixed the test's magnitude, not the underlying
   `evidential_forward` math, which was always correct.
2. **Test-fixture bug (mine)** — an early integrity-check test fixture
   generated images using a per-class loop counter for color, causing
   accidental cross-class pixel-identical "duplicate" images. The duplicate
   detector correctly caught this; the fixture generator was the bug, fixed
   by using a global monotonic counter.
3. **REAL pre-existing project bug** — `UnifiedCropDataset.__getitem__` and
   `get_raw_sample()` read samples from each sub-dataset's *original*
   `.samples` list, bypassing any overlay (e.g. severity annotations)
   applied to the flattened `_all_samples` list. Since `Sample` is a frozen
   dataclass, the severity overlay (`attach_severity_labels`) produces NEW
   objects via `dataclasses.replace` — the original list is never mutated,
   so the overlay was being silently ignored at the exact point it
   mattered. **Fixed**: both methods now read from `_all_samples`
   consistently. Caught by writing an explicit round-trip test, not by
   inspection alone.
4. **REAL bug caught before execution (would have been a runtime
   `ImportError`)** — `training/trainer.py` initially imported
   `build_evidential_loss` from `losses/evidential_loss.py`. Inspecting the
   actual pre-existing file (rule: never assume file contents) showed the
   real factory function is named `build_loss`. Fixed the import and call
   site.
5. **REAL bug caught before execution (would have been a runtime
   `TypeError`)** — `training/trainer.py`'s training loop initially called
   `self.loss_fn(outputs, targets)`, but
   `MultiTaskEvidentialLoss.forward(self, outputs, targets, epoch)` requires
   a third positional `epoch` argument (used for KL annealing). Fixed by
   passing `epoch` through from the training loop, and also fixed a related
   omission: `self.model(images)` wasn't passing `discount_severity`
   explicitly from config, which matters for the A3/A4 ablations later.
6. **Test-assertion bug (mine)**, caught during Phase 2 unified-dataset
   testing — an assertion about `resolved_count()` conflated CSV-level
   "has a resolved_severity value" with "successfully attached after the
   crop/disease mismatch check," which are different things by design
   (mismatch check happens in `attach_severity_labels`, not
   `resolved_count()`). Fixed the assertion, not the code, after confirming
   the code's behavior was actually correct and intentional.

**No bug has ever been found in**: `models/evidential.py`'s core math (18/18
independently verified), `training/early_stopping.py`,
`training/lr_schedule.py`, `training/checkpoint_manager.py`'s bookkeeping —
all were correct on first real execution.

## 10. Pending Tasks

In rough priority order:
1. **Finish Phase 3 as most recently scoped** (Section 6's "not yet built"
   list) — rename files, build `callbacks.py`/`logger.py`/`config.py`,
   build `evaluation/`, build the 10-point verification test suite
   (executing everything torch-free that can be, marking the rest
   unverified), write `research/threats_to_validity.md`.
2. Get real torch/timm/albumentations installed in whatever environment
   continues this project, and actually run: `tests/test_evidential.py`,
   `tests/test_fusion.py`, `models/agrivisionnet.py`'s shape check, all of
   `training/trainer.py`'s logic, and the full `pytest tests/` suite.
   **Nothing torch-dependent in this entire project has ever been executed.**
3. **Decide and implement A1 (shared-fusion baseline) and A5 (non-evidential
   baseline)** — currently `training/train.py` raises `NotImplementedError`
   for both; they require new model code (`SharedReliabilityGate` class,
   a parallel non-evidential head/gate path) which is an **architecture
   change requiring explicit owner approval** per the freeze.
4. Build the 3 single-backbone baseline training paths (ResNet50,
   EfficientNet-B0-only, Swin-Tiny-only) — `training/train.py` currently
   raises `NotImplementedError` for any `--model` other than
   `agrivisionnet`.
5. Get a real dataset downloaded and verify it against the documented
   (currently unverified) folder-structure assumptions in
   `datasets/plantvillage.py`, `paddy_doctor.py`, `rice_disease.py`.
6. Perform the actual manual severity annotation per
   `research/severity_annotation_protocol.md` — zero annotation has
   happened; `datasets/severity_annotations.py` has no real CSV to load yet.
7. Phases 4 (baselines/ablations) through 11 (IEEE paper) — not started at all.

## 11. Known Limitations

- **No dataset has ever been downloaded in this project.** All three
  loaders are built against documented public conventions, explicitly
  confidence-tagged (`datasets/README.md`): PlantVillage moderate-high
  confidence, Paddy Doctor moderate, "Rice Disease" **low confidence and
  genuinely ambiguous** (at least two different public datasets share this
  name).
- **No severity ground truth exists yet** — every sample's severity is
  `None`/`MISSING_LABEL(-1)` until real annotation happens.
- **Nothing torch-dependent has ever been executed** in any environment
  used for this project. Every claim about Phase 1/3 model/training code
  correctness beyond what's covered by `scripts/verify_core_math_numpy.py`
  and the isolated pure-Python training utilities is "written to be
  correct," not "verified to be correct."
- **`albumentations` has never been installed** — `datasets/transforms.py`'s
  `build_train_transform`/`build_eval_transform` are unverified; only the
  torch-free MixUp/CutMix math in that file has been tested.
- No literature search tool was ever available during this project's
  design phase — all citations (TMC, MMTM, AFF, Mobile-Former, EDL, QMF,
  RCML, etc.) are from model training-data memory, explicitly flagged at
  the time as needing verification, especially QMF/RCML (lower confidence,
  possibly imprecise on exact venue/year/mechanism) and anything from
  2023–2026 generally.

## 12. Research Decisions (chronological, with rationale)

1. **Rejected** an initial "custom Adaptive Feature Fusion Module"
   (channel+spatial attention, bidirectional cross-attention, dynamic
   gating) after a self-review survey showed it recombines SE (2018), CBAM
   (2018), Mobile-Former's bridge (2022), and AFF/MMTM's gating (2020/2021)
   with no real delta.
2. **Adopted** evidential/uncertainty-based fusion as the new direction,
   with TMC/ETMC identified as the primary baseline to differentiate
   against.
3. **Ran a hostile-reviewer-style gap analysis** across 5 candidate
   directions (uncertainty-aware fusion alone; task-aware fusion; hierarchical
   evidential learning; the combination; and open brainstorming of
   alternatives like causal fusion, prototype-guided reasoning, curriculum
   learning — all considered and rejected as infeasible/too-thin/too-risky
   for the timeline).
4. **Froze** on the combination: task-aware gating + hierarchical
   discounting, explicitly acknowledged as an **incremental** contribution,
   appropriate for a Scopus IEEE conference, not a top-tier venue.
5. **Discovered** the severity-label gap only after the architecture was
   frozen — decided (owner's explicit call) to resolve via **manual
   annotation only**, never algorithmic derivation as primary supervision;
   a lesion-area proxy may exist later as an auxiliary experiment only.

## 13. Engineering Rules (established across the project, still binding)

1. Never claim code is verified unless it has actually been executed.
2. Distinguish explicitly: *executed and verified* vs. *written but
   unverified (environment limitation)* — every deliverable message in this
   project has done this.
3. Never fabricate benchmarks, training results, or passing tests.
4. **Inspect existing files before extending them** — do not assume
   contents from memory. This rule caught real bugs (Section 9, items 4-5).
5. Document whether a discovered bug is a pre-existing project bug or a bug
   in the reviewer's/tester's own new code (Section 9 does this for every
   bug found).
6. Every module should carry literature references where appropriate, with
   confidence-level caveats where the citation wasn't independently
   verifiable (no search tool available during this project).
7. **Architecture is frozen.** No model changes without explicit owner
   approval (Section 19).

## 14. Reproducibility Decisions

Full detail in `research/reproducibility.md` (every field tagged
MEASURED/PINNED/PENDING — do not add a number to that file without one of
these tags). Key points: seed = 42 primary, multi-seed protocol is
`{42,43,44,45,46}` fixed in advance (not re-rolled); `cudnn.deterministic=True`
traded off explicitly against throughput; split indices are persisted to
disk, not just re-derived from a seed; every run must write a
`resolved_config.yaml` for full config-to-result traceability
(`CheckpointManager.save_resolved_config` implements this).

## 15. Dataset Assumptions

Full detail in `datasets/README.md`. Summary:
- **PlantVillage**: `<root>/{Crop}___{Disease}/*.jpg` (triple underscore).
  Moderate-high confidence.
- **Paddy Doctor**: `<root>/train_images/<label>/*.jpg` + optional
  `train.csv`. Moderate confidence. Single-crop (rice, hardcoded correctly,
  not a placeholder).
- **"Rice Disease"**: genuinely ambiguous name, two layouts supported
  (split-subfolder and flat-subfolder), loader tries both. **Low
  confidence — needs the owner to confirm the actual source before trusting
  this loader.**
- Splits are **always re-computed from scratch** (`datasets/splitting.py`),
  never inherited from a dataset's own bundled train/test split — reasons
  documented in `datasets/README.md` (fair 4-way baseline comparison;
  avoiding pre-augmentation-then-split leakage in some public mirrors).

## 16. Annotation Protocol Summary

Full doc: `research/severity_annotation_protocol.md`. Key points:
- **4-level ordinal scale**: none/healthy, mild (>0–10% leaf area), moderate
  (>10–25%), severe (>25%) — a deliberate simplification of Horsfall-Barratt,
  stated as such.
- **8–10 disease classes selected** (not all 38+ PlantVillage classes),
  40–60 images each, ~350–600 images total — small enough for **full double
  annotation** rather than partial-overlap sampling.
- **Two independent annotators**, blind to each other; adjacent
  disagreements go to a third adjudicator; non-adjacent disagreements
  (≥2 grades apart) trigger re-examination, treated as a likely
  protocol/image-quality issue.
- **Quadratic-weighted Cohen's κ** (ordinal-appropriate), computed on RAW
  pre-adjudication labels only (never on reconciled labels, which would
  trivially inflate agreement). κ < 0.40 should trigger protocol
  re-calibration, not silent acceptance.
- **CSV schema** is fixed (`REQUIRED_COLUMNS` in
  `datasets/severity_annotations.py`, must match exactly): `image_filepath,
  source_dataset, crop_name, disease_name, annotator_1_id,
  annotator_1_severity, annotator_2_id, annotator_2_severity,
  adjudicator_id, resolved_severity, partial_leaf, exclude_multi_disease,
  exclusion_reason, notes`.
- **No annotation has been performed.** The file this schema describes does
  not exist. Pipeline code is ready and tested against a synthetic
  schema-test fixture only.

## 17. Current TODO List

(Same as Section 10, restated as an actionable checklist)

- [ ] Rename `training/lr_schedule.py`→`scheduler.py`,
      `checkpoint_manager.py`→`checkpoint.py`, `train.py`→`cli.py`
- [ ] Build `training/callbacks.py` (NaN guard + gradient monitor, decision
      logic separated from tensor extraction per the established pattern)
- [ ] Build `training/logger.py` (TensorBoard + CSV facade)
- [ ] Build `training/config.py` (extract config loading from CLI)
- [ ] Build `evaluation/metrics.py`, `calibration.py`, `confusion.py`
      (sklearn/numpy — actually testable in a no-torch sandbox)
- [ ] Build the 10-point training verification test suite, executing
      everything torch-free (seed reproducibility, scheduler-resume math,
      NaN/zero-gradient decision logic) and marking the rest unverified
- [ ] Write `research/threats_to_validity.md`
      (Internal/External/Construct/Conclusion validity, per owner's bullets)
- [ ] Get a real torch/timm/albumentations environment and run everything
      that has never been executed
- [ ] Resolve A1/A5 ablation architecture additions (needs owner approval first)

## 18. Next Immediate Task

**Build `training/callbacks.py` first.** It's a dependency for both
`logger.py` (CSV logging facade wraps callbacks) and the NaN-detection /
zero-gradient tests the owner explicitly requested. Follow the
`checkpoint_manager.py` pattern exactly: separate pure-Python decision
logic (e.g. `NaNGuard.check(loss_value: float) -> bool`,
`GradientMonitor.check(param_norms: Dict[str, float]) -> List[str]`) from
any actual tensor extraction, so the decision logic can be executed and
verified in this torch-free sandbox even though the full integration into
`Trainer` cannot be.

## 19. Files That Should Never Be Modified Without Approval

- **All of `models/`** (architecture is frozen — Section 3)
- **`models/fusion/task_hierarchy_fusion.py`** specifically — this IS the
  paper's claimed contribution; any change here invalidates the frozen
  novelty argument in `research/design_rationale.md`
- **`losses/evidential_loss.py`** — the loss formulation is part of the
  frozen design (KL annealing, per-task evidential loss, auxiliary
  reliability supervision)
- **`configs/config.yaml`'s architecture-defining sections** (`backbone`,
  `fusion`, `heads`, `loss.strategy`) — hyperparameter *values* within the
  frozen structure are more negotiable than the structure itself, but check
  first
- **`research/design_rationale.md`'s stated non-claims** — do not let a
  future draft quietly upgrade the novelty claim beyond what's stated there

Everything in `datasets/`, `training/`, `evaluation/` is Phase 2/3
engineering and can be refactored/extended freely as long as it doesn't
change what data reaches the model or what the model computes.

## 20. Implementation Details Another Engineer Would Need

- **`Sample` is a frozen dataclass.** You cannot mutate one in place —
  `dataclasses.replace(sample, severity_name=...)` is the pattern, already
  used in `datasets/severity_annotations.py`. Any new code that needs to
  modify a `Sample` post-construction must use this pattern, not attempt
  in-place mutation (which will raise `FrozenInstanceError`).
- **`MISSING_LABEL = -1`** is defined once in `datasets/base_dataset.py` and
  must stay consistent with `losses/evidential_loss.py`'s `ignore_index`
  default. If either changes, the other must be updated in the same commit
  — there is no shared config value enforcing this consistency currently,
  which is itself a latent risk worth fixing (consider promoting this to a
  single config-driven constant in a future refactor).
- **`UnifiedCropDataset.__init__`'s `label_spaces` parameter is not
  optional in practice for val/test splits** — passing `None` refits fresh
  label spaces from whatever data is given, which is only correct for the
  train split. Always pass `label_spaces=train_ds.label_spaces()` for
  eval/test dataset construction, or you get train/val label leakage
  (train sees classes val doesn't, or vice versa, silently).
- **`MultiTaskEvidentialLoss.forward` requires `epoch` as a third
  positional argument** for KL annealing — this was missed once already in
  `trainer.py` (Section 9, bug 5) and is an easy omission to reintroduce in
  any new training/eval script.
- **`AgriVisionNet.forward`'s `discount_severity` parameter defaults to
  `True`** — for the A2/A3 ablations (task-aware-only vs.
  hierarchical-only), this must be passed explicitly from
  `cfg["loss"]["discount_severity_by_disease_uncertainty"]`, not left at
  the default, or the ablation silently doesn't do what its name claims.
- **timm's Swin `features_only=True` output format has changed across
  versions** (NCHW vs NHWC vs flat token sequence) —
  `models/backbone.py:SwinBranch.forward` defensively normalizes all three,
  but this defensive code has **never been exercised against a real timm
  install**. Verify the actual installed timm version's output shape before
  trusting this.
- **No dataset in this project provides severity ground truth as of
  handoff** — any new training run will have a severity task loss of
  approximately zero effective signal (everything masked by
  `ignore_index=-1`) until real annotation happens. This is expected, not a
  bug, but will look like a stalled/frozen severity metric if not
  anticipated.
- **This entire project has been built in a sandbox with no GPU, no
  network access, and no `torch`/`timm`/`albumentations`/`pytest`
  installed.** The very first action in any continuation should be
  confirming what IS installed in the new environment before assuming
  anything works — do not assume the next environment matches this one.

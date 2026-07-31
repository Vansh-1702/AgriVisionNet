# PROJECT_TREE.md — AgriVisionNet

Generated directly from the actual filesystem at packaging time (not
reconstructed from memory) — see `PROJECT_HANDOFF.md` Section 4/7 for the
narrative version with fuller rationale per file. `.gitkeep` files mark
directories reserved for phases that haven't started yet (Phases 4–9); they
contain no content.

```
AgriVisionNet/
├── backend/                                  # empty -- Phase 8 (FastAPI), not started
│   └── .gitkeep
├── configs/
│   └── config.yaml                           # single source of truth: backbone/fusion/heads/loss/data/training hyperparams
├── data/                                      # empty -- no real dataset ever downloaded in this project
│   └── .gitkeep
├── datasets/                                  # Phase 2 -- DONE, torch-free logic executed & passing
│   ├── README.md                             # documented layout assumptions, severity gap, verification status
│   ├── __init__.py                           # registers all loaders via import side-effect
│   ├── base_dataset.py                       # Sample (frozen dataclass), BaseCropDataset, MISSING_LABEL=-1
│   ├── integrity.py                          # corrupt/zero-byte/duplicate/missing-file checks, imbalance ratio
│   ├── label_space.py                        # cross-dataset label unification (LabelSpace, normalize_label)
│   ├── paddy_doctor.py                       # PaddyDoctorDataset loader (moderate confidence layout)
│   ├── plantvillage.py                       # PlantVillageDataset loader (moderate-high confidence layout)
│   ├── registry.py                           # @register_dataset decorator, build_dataset() factory
│   ├── rice_disease.py                       # RiceDiseaseDataset loader (LOW confidence -- name is ambiguous)
│   ├── severity_annotations.py               # manual annotation CSV loader/overlay/Cohen's kappa
│   ├── splitting.py                          # stratified_split(), DatasetSplit (persisted to disk)
│   ├── statistics.py                         # compute_dataset_statistics()
│   ├── transforms.py                         # Albumentations pipeline builder + torch-free MixUp/CutMix math
│   └── unified.py                            # UnifiedCropDataset -- the actual training-facing object
├── evaluation/                                # empty -- Phase 3 (metrics/calibration/confusion), not built yet
│   └── .gitkeep
├── explainability/                            # empty -- Phase 6 (Grad-CAM, attention rollout), not started
│   └── .gitkeep
├── frontend/                                  # empty -- Phase 9 (React+Tailwind), not started
│   └── .gitkeep
├── losses/
│   └── evidential_loss.py                    # MultiTaskEvidentialLoss, build_loss() -- epoch is a REQUIRED forward() arg
├── models/                                    # Phase 1 -- FROZEN, math-verified via numpy (18/18), nn.Module never run
│   ├── fusion/
│   │   ├── attention_blocks.py               # ChannelAttention (SE), SpatialAttention (CBAM) -- standard, not novel
│   │   └── task_hierarchy_fusion.py          # PerTaskReliabilityGate, TaskHierarchyEvidentialFusion -- THE claimed contribution
│   ├── agrivisionnet.py                      # AgriVisionNet (assembled model), build_model()
│   ├── backbone.py                           # EfficientNet-B0 + Swin-Tiny dual branch (standard, not novel)
│   ├── evidential.py                         # DirichletOutput, EvidentialHead, ds_discount(), kl_annealing_coefficient()
│   └── heads.py                              # MultiTaskEvidentialHeads
├── outputs/                                   # empty -- Phase 10 (generated figures/tables), not started
│   └── .gitkeep
├── recommendation/                            # empty -- Phase 7 (treatment engine), not started
│   └── .gitkeep
├── research/
│   ├── design_rationale.md                   # literature positioning vs TMC/ETMC/MMTM/AFF/Mobile-Former, ablation plan, explicit non-claims
│   ├── experiments.md                        # baselines, metrics, A1-A7 ablations (A1/A5 flagged as needing new code), stats protocol
│   ├── reproducibility.md                    # every field tagged MEASURED/PINNED/PENDING -- no fabricated numbers
│   └── severity_annotation_protocol.md       # scale, criteria, guidelines, volume, kappa protocol, CSV schema
├── scripts/
│   └── verify_core_math_numpy.py             # torch-free EDL/DS-discount math verification -- EXECUTED, 18/18 pass
├── tests/
│   ├── test_datasets/                        # Phase 2 -- EXECUTED & PASSING (torch-free)
│   │   ├── conftest.py                       # synthetic (non-experimental) fixture directories
│   │   ├── test_base_dataset.py
│   │   ├── test_integrity.py
│   │   ├── test_label_space.py
│   │   ├── test_paddy_doctor.py
│   │   ├── test_plantvillage.py
│   │   ├── test_rice_disease.py
│   │   ├── test_severity_annotations.py
│   │   ├── test_splitting.py
│   │   ├── test_statistics.py
│   │   ├── test_transforms.py
│   │   └── test_unified.py
│   ├── test_training/                        # Phase 3 -- PARTIAL, executed & passing for what's built so far
│   │   ├── test_checkpoint_manager.py
│   │   ├── test_early_stopping.py
│   │   └── test_lr_schedule.py
│   ├── test_evidential.py                    # Phase 1 -- requires torch+timm, UNVERIFIED in this sandbox
│   └── test_fusion.py                        # Phase 1 -- requires torch+timm, UNVERIFIED in this sandbox
├── training/                                  # Phase 3 -- IN PROGRESS, see PROJECT_HANDOFF.md Section 6
│   ├── checkpoint_manager.py                 # CheckpointManager -- pure bookkeeping, tested; will be renamed checkpoint.py (not yet done)
│   ├── early_stopping.py                     # EarlyStopping -- tested (mode min/max, min_delta, resume round-trip)
│   ├── lr_schedule.py                        # cosine_warmup_multiplier() -- tested; will be renamed scheduler.py (not yet done)
│   ├── seeding.py                            # set_seed(), seed_worker()
│   ├── train.py                              # CLI (resolve_config tested; main() untested); will be renamed cli.py (not yet done)
│   └── trainer.py                            # Trainer class -- written completely, NEVER EXECUTED (needs torch)
├── weights/                                   # empty -- will hold checkpoints once Phase 3 training actually runs
│   └── .gitkeep
├── PROJECT_HANDOFF.md                        # full narrative handoff -- read this first in a new conversation
├── PROJECT_TREE.md                           # this file
├── README.md                                 # top-level status tracker
└── requirements.txt                          # pinned-with->= dependency list; see reproducibility.md for exact-pin guidance
```

## Quick status legend

- **DONE** = built and either math-verified (numpy) or executed against real/synthetic data in this sandbox.
- **IN PROGRESS** = partially built; see `PROJECT_HANDOFF.md` Section 6 for the exact resume point.
- **empty / not started** = directory exists as a placeholder for a phase that hasn't begun.

## What's verified vs. not, at a glance

Nothing that imports `torch` has ever been executed in any environment used
for this project (no GPU/network access — see `README.md`). Everything
else — Phase 2's full data pipeline, the pure-math parts of Phase 1
(`scripts/verify_core_math_numpy.py`), and Phase 3's non-torch utilities
(`early_stopping.py`, `lr_schedule.py`, `checkpoint_manager.py`,
`train.py`'s `resolve_config()`) — has been actually run, with results
reported honestly, including several real bugs found and fixed along the
way (catalogued in `PROJECT_HANDOFF.md` Section 9).

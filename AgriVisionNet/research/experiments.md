# Experiment Plan

**Status: spec for Phases 3/4/10.** No experiment below has been run — this
document defines what "done" means for those phases and the exact command
each row/figure maps to (see `research/reproducibility.md` Section 8). Do not
fill in result numbers here speculatively; this file's job is to pin down
*what will be measured and how*, before any number exists.

---

## 1. Primary baselines

| Model | Role | Params source |
|---|---|---|
| **ResNet50** (He et al., CVPR 2016) | Classic CNN baseline | `torchvision.models.resnet50` or `timm` |
| **EfficientNet-B0** (Tan & Le, ICML 2019) | Modern efficient CNN baseline; also our CNN branch in isolation | `timm.create_model('efficientnet_b0')` |
| **Swin-Tiny** (Liu et al., ICCV 2021) | Transformer baseline; also our Swin branch in isolation | `timm.create_model('swin_tiny_patch4_window7_224')` |
| **AgriVisionNet (ours)** | Task- and Hierarchy-Aware Evidential Fusion | `models/agrivisionnet.py` |

All four trained on identical data splits, identical augmentation, identical
optimizer/schedule family (differences in learning rate *magnitude* are
allowed per-architecture if justified by a short LR-range-test, but the
*schedule shape* — cosine warmup — must be held constant, or the comparison
isn't clean). Each of the single-backbone baselines (ResNet50, EffNet-B0,
Swin-Tiny) gets a plain softmax head trained with standard cross-entropy on
the **disease** label only, since they can't natively do our multi-task
output — this is stated explicitly as a design decision, not hidden: it
means the baseline comparison is fair for disease classification accuracy,
but the multi-task and uncertainty-quality metrics (Section 2) are
necessarily reported for AgriVisionNet only, with the ablations (Section 3)
carrying the burden of showing which of *our own* components drive those
numbers, rather than claiming the baselines "lose" at something they were
never designed to do.

## 2. Evaluation metrics

| Metric | What it measures | Applies to |
|---|---|---|
| Accuracy | Top-1 classification accuracy | All 4 models, disease task; crop/severity for ours only |
| Precision / Recall / F1 (macro-averaged, given class imbalance is likely) | Per-class performance, robust to imbalance | All 4 models |
| AUROC (one-vs-rest, macro-averaged) | Ranking quality across classes | All 4 models |
| **ECE — Expected Calibration Error** | Whether predicted confidence matches actual correctness rate | All 4 models (softmax confidence for baselines; Dirichlet expected probability for ours) — this is the metric that most directly tests whether the evidential mechanism buys anything |
| Inference time (ms/image, batch=1, and throughput at batch=32) | Deployment feasibility (FastAPI serving, Phase 8) | All 4 models, same hardware, same precision (fp32 and fp16/AMP both reported) |
| Parameters (millions) | Model size | All 4 models |
| FLOPs (GFLOPs at 224×224 input) | Compute cost, hardware-independent | All 4 models — use `fvcore` or `thop`, report which tool was used (they don't always agree) |

**Additional metrics specific to the uncertainty claim** (not in the user's
original list, but required to actually validate what this project claims —
flagging the addition rather than silently adding scope):
- **AUROC for OOD/unknown-disease rejection** (in-distribution vs. held-out-class test, per `research/design_rationale.md`'s cross-dataset OOD protocol) — the evidential uncertainty's actual job.
- **Reliability diagram bin-wise gap** (visual complement to scalar ECE, see Section 4 figures).

## 3. Ablation studies

All ablations use **AgriVisionNet only** (baselines are not ablated — they
have no fusion/hierarchy mechanism to ablate). Each row states the exact
config/code change and what it isolates.

| ID | Name | Change from full model | Isolates |
|---|---|---|---|
| **A1** | Shared Fusion | Replace `PerTaskReliabilityGate` (3 separate gate MLPs) with **one shared gate** used identically for all 3 task heads | Whether task-*specific* gating matters at all (this is also the closest reproduction of an MMTM/AFF-style baseline within our own codebase — **not yet implemented**, requires a new `SharedReliabilityGate` class alongside the existing `PerTaskReliabilityGate` in Phase 4, config flag `fusion.gate_mode: shared \| per_task`) |
| **A2** | Task-aware Fusion | Full model, `gate_mode: per_task`, discounting **off** | The task-aware mechanism in isolation, without the hierarchy mechanism confounding it |
| **A3** | Hierarchical Discounting | `gate_mode: shared`, discounting **on** | The hierarchy mechanism in isolation, without task-aware gating confounding it |
| **A4** | Task-aware + Hierarchical (full model) | `gate_mode: per_task`, discounting **on** — this is the frozen architecture as built | The combined claim; compared against A1/A2/A3, shows whether the two mechanisms are additive, redundant, or interacting |
| **A5** | Without Evidential Learning | Replace all `EvidentialHead` instances (task heads **and** the Stage-3 auxiliary branch-reliability heads) with plain softmax + cross-entropy; Stage-4 gate then conditions on raw pooled-feature statistics (MMTM-style) instead of `1 - uncertainty`, since there is no uncertainty left to condition on — **requires a parallel non-evidential head/gate implementation**, not yet built | Whether EDL specifically (vs. any gated fusion) is doing the work, or whether a non-evidential gated-fusion baseline gets similar accuracy — this is the single most important ablation for defending the paper's core claim |
| **A6** | Without Explainability | Disable Grad-CAM / attention-rollout generation at inference | **Not an accuracy ablation** — explainability (Phase 6) is a post-hoc, inference-only visualization layer that does not feed back into training or predictions. This row exists to document that removing it changes *no metric in Section 2*, and its "result" is qualitative (a note in the paper, not a table row) — do not report a fabricated accuracy delta for this row |
| **A7** | Without Unknown Disease Detection | Disable the inference-time uncertainty-threshold rejection gate (Phase 5) | Also not a Section-2 in-distribution accuracy ablation. Measured instead on the OOD protocol only: with A7 off, report the false-accept rate (fraction of true out-of-scope/unknown-class images the system confidently labels as a known disease instead of rejecting) |

**Cost note for Phase 4 planning:** A1–A5 require 5 model variants × 5 seeds = 25 runs minimum (A6/A7 are inference-time toggles on the already-trained A4/full model, no extra training runs needed).

## 4. Statistical validation

- **5 seeds per configuration**: `{42, 43, 44, 45, 46}`, fixed in advance (see `research/reproducibility.md` Section 2) — not re-rolled until a favorable result appears.
- **Report mean ± standard deviation** for every metric in Section 2, for every model in Section 1 and every ablation in Section 3.
- **Significance testing**: given 5 runs per condition (small-sample, likely non-normal), use a **paired Wilcoxon signed-rank test** (pairing by seed) between AgriVisionNet (A4) and each baseline/ablation, rather than an independent-samples t-test, which assumes normality that 5 samples cannot establish. Report the exact p-value and note explicitly that n=5 limits statistical power — do not overstate significance from a small sample. If reviewers push back, a bootstrap confidence interval (resample the per-image predictions, not the 5 run-level numbers) over a single run is a reasonable supplementary analysis, but is not a substitute for the multi-seed comparison above.
- **Multiple-comparisons caution**: A1–A7 plus 3 baselines is 10 comparisons against the full model; if reporting p-values across all of them, apply a Holm-Bonferroni correction and say so, rather than presenting 10 uncorrected p-values as independently significant.

## 5. Expected figures

| Figure | Source data | Script (Phase 10, not yet built) |
|---|---|---|
| ROC curves (one-vs-rest, overlaid per model) | Section 1 eval | `evaluation/plot_roc_pr.py` |
| Precision–Recall curves | Section 1 eval | `evaluation/plot_roc_pr.py` (same script, second axis) |
| Confusion matrix (disease task, best-seed run for each model) | Section 1 eval | `evaluation/plot_confusion_matrix.py` |
| Calibration curve (predicted confidence vs. empirical accuracy, binned) | Section 2 ECE computation | `evaluation/plot_calibration.py` |
| Reliability diagram (bar-chart form of the calibration curve, standard in the EDL/calibration literature — distinct from the line-plot calibration curve, both are conventionally shown) | Same underlying bins as above | `evaluation/plot_calibration.py` (second plot) |
| Grad-CAM examples (correct-confident, correct-uncertain, incorrect-confident, incorrect-uncertain — 4 qualitative categories, not just "some examples") | `explainability/gradcam.py` (Phase 6, not yet built) output | `explainability/gradcam_report.py` |
| Training curves (loss + per-task accuracy vs. epoch, train and val) | TensorBoard logs from Phase 3 | `evaluation/plot_training_curves.py` |
| Ablation chart (bar chart, A1–A7 vs. full model, per key metric) | Section 3 results table | `evaluation/plot_ablation_chart.py` |

Every figure must be regeneratable from its single script call in `research/reproducibility.md` Section 8, from data traceable to a specific `run-name`'s `resolved_config.yaml` — no figure should exist that can't be traced back to the exact run and config that produced it.

## 6. Explicit scope boundary for this document

This plan does not include hyperparameter search results, since Phase 3
(training framework) doesn't exist yet — LR/weight-decay/architecture-width
sweeps, if any, belong in a separate `research/hyperparameter_search.md` once
Phase 3 makes them possible to run, not invented here.

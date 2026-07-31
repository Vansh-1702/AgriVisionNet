# Design Rationale — AgriVisionNet Phase 1 (Frozen Architecture)

**Architecture:** Task- and Hierarchy-Aware Evidential Fusion for Hybrid CNN–Swin
Multi-Task Crop Disease Analysis

**Positioning:** This is an **integrated systems contribution**. No individual
mechanism below is claimed as new. The claim is the specific combination —
evidential-reliability-conditioned, per-task feature fusion, plus
Dempster-Shafer hierarchical discounting across a crop→disease→severity task
structure — applied to hybrid CNN-Transformer crop disease diagnosis.

> ⚠️ Citations below are from training-data memory; no live literature search
> was available in this environment when they were compiled. Verify exact
> venue/year/mechanism for each before finalizing a paper's related-work
> section — TMC and AFF/MMTM/Mobile-Former are moderate-to-high confidence;
> QMF and RCML were flagged as lower-confidence recollections throughout
> this project's design discussion and should be checked first.

---

## Component-by-component justification

### 1. Dual backbone (`models/backbone.py`) — NOT part of the novelty claim
- **EfficientNet-B0**: Tan & Le, ICML 2019. Local/texture inductive bias, matches
  the fine-grained, translation-invariant nature of lesion texture.
- **Swin Transformer-Tiny**: Liu et al., ICCV 2021. Global context via
  shifted-window self-attention; hierarchical pyramid keeps spatial resolution
  compatible with EfficientNet's final feature map, avoiding ViT's single-scale
  reshaping problem.
- Standard practice, cited as such, no claim made here.

### 2. Channel/spatial attention (`models/fusion/attention_blocks.py`) — NOT novel
- **SENet** (Hu, Shen, Sun, CVPR 2018) — channel attention.
- **CBAM** (Woo, Park, Lee, Kweon, ECCV 2018) — spatial attention.
- Used as standard per-branch recalibration before the task-aware gating below.

### 3. Evidential Deep Learning core (`models/evidential.py`) — NOT novel
- **Sensoy, Kaplan, Kandemir, NeurIPS 2018** — Dirichlet-evidence head,
  type-II MLE loss, annealed KL regularizer. All formulas reused as published.
- **Malinin & Gales, NeurIPS 2018** (Dirichlet Prior Networks) — documented
  alternative that separates aleatoric/epistemic uncertainty more explicitly;
  not implemented, to keep one consistent formulation across the project.
  Worth an appendix discussion if reviewers ask why EDL over Prior Networks.
- **Shafer, 1976** — the Dempster-Shafer discounting operator (`ds_discount`).
  This is a classical, general operator; our use of it for a *hierarchical
  task* relationship (not multi-source combination) is part of our delta,
  not the operator itself.

### 4. The actual claimed delta (`models/fusion/task_hierarchy_fusion.py` + `models/heads.py`)

**Nearest prior work and how we differ, stated explicitly (not left for a
reviewer to notice):**

| Prior work | What it does | Shared with ours | Genuinely different |
|---|---|---|---|
| **TMC** (Han et al., ICLR 2021) | Per-view Dirichlet, DS-combined at decision level, single-task | Dirichlet/EDL core, DS-family operator | Feature-level (not decision-level) fusion; multi-task (not single-task); DS *discounting* (not *combination*) |
| **ETMC** (TPAMI extension of TMC) | Adds a joint pseudo-view | Same as TMC | Same delta as above |
| **MMTM** (Joze et al., CVPR 2020) | Per-sample gate from pooled multimodal *feature* stats | Feature-level gated fusion pattern | Gate conditioned on evidential *reliability* (1-uncertainty), not raw feature magnitude; separate gate *per task* |
| **AFF/iAFF** (Dai et al., WACV 2021) | Channel-attention-weighted sum of two feature maps | Same feature-level fusion family | Same two deltas as MMTM row |
| **Mobile-Former** (Chen et al., CVPR 2022) | Bidirectional cross-attention bridge, general-purpose backbone | Dual-branch bridging concept | Different purpose (efficiency vs. reliability); we don't use cross-attention, we use reliability-gated mixing |
| **QMF** (~2023, low-confidence recollection) | Per-modality scalar "quality" reweighting | Reliability/quality-conditioned dynamic fusion | We use full Dirichlet evidence + explicit task hierarchy, not a single scalar quality score — **verify this baseline's actual mechanism before citing** |
| **RCML** (~2024, low-confidence recollection) | Explicit conflict handling between views (beyond flat TMC) | Evidential multi-view combination, post-TMC | Signals the space is more contested than TMC alone suggests — **verify before citing** |

**Two independently ablatable mechanisms constitute the contribution:**
1. **Task-aware gating** — a separate reliability-conditioned gate per task
   head (crop/disease/severity), vs. one shared gate (MMTM/AFF-style).
2. **Hierarchy-aware discounting** — severity's Dirichlet output is
   Dempster-Shafer-discounted by the disease head's own uncertainty,
   formalizing "don't trust severity if disease identity itself is unsure."
   None of TMC/ETMC/MMTM/AFF/Mobile-Former model an explicit task hierarchy.

### 5. Why evidential heads for ALL three tasks, not just disease
Required, not stylistic: Stage 5 discounting needs severity's output to
*be* a Dirichlet (something `ds_discount` can operate on). A softmax severity
head would give the discounting operator nothing to act on. Crop is made
evidential too for consistency and because a future crop-level "unknown crop"
rejection path (not yet built) would need the same mechanism.

### 6. Why auxiliary per-branch evidential heads exist (Stage 3 of fusion)
Branch reliability (used to build the Stage 4 gate) has to come from
*somewhere* trained. Two lightweight evidential heads run on each branch's
pooled features (disease label space only, the hardest/most informative
task) and are supervised with a small auxiliary loss term
(`losses/evidential_loss.py:MultiTaskEvidentialLoss`, `aux_weight=0.3`).
Without this supervision, Stage 4's gate would condition on effectively
untrained, near-random reliability estimates.

---

## Required ablation studies (Phase 4/Phase 10 will report these)

| # | Ablation | Isolates |
|---|---|---|
| 1 | Shared gate (one MLP, all 3 tasks) vs. per-task gate | The "task-aware" claim specifically |
| 2 | With vs. without Stage 5 hierarchical discounting | The "hierarchy-aware" claim specifically |
| 3 | Gate conditioned on evidential reliability vs. raw pooled features (MMTM-style) | The "evidential-conditioning" claim vs. "having a gate at all" |
| 4 | Full architecture vs. TMC-style decision-level fusion (same backbones) | Feature-level vs. decision-level fusion, holding backbone/data fixed |
| 5 | EDL heads vs. plain softmax heads + separate MC-Dropout/Deep-Ensemble uncertainty | Whether EDL's single-pass uncertainty is competitive with sampling-based alternatives, at inference-cost parity |
| 6 | With vs. without auxiliary branch-reliability supervision (`aux_weight=0` vs. `0.3`) | Whether the reliability signal is actually learned, not vacuous |
| 7 | Fixed task-loss weights vs. learned uncertainty-weighting on top of evidential losses | Whether stacking Kendall-style task weighting adds anything beyond evidential losses alone |

## Explicit non-claims (to state plainly in the paper's related-work section)

- We do **not** claim to have invented CNN-Transformer hybrid architectures,
  Evidential Deep Learning, Dempster-Shafer theory, channel/spatial attention,
  or gated multimodal fusion in general.
- We do **not** claim state-of-the-art accuracy is the point — the
  contribution is the uncertainty-aware *systems design*, evaluated primarily
  through calibration (ECE), OOD/unknown-disease rejection quality, and the
  ablations above, not solely top-line classification accuracy.
- Honest self-assessment: this is an **incremental** contribution by design
  (per the frozen decision recorded in project discussion), appropriate for a
  Scopus-indexed IEEE conference paper, not a top-tier architecture venue.

# Severity Annotation Protocol

**Status: frozen decision, per user approval.** Severity labels for this
project are a manually-created research dataset — never fabricated,
derived, or heuristically generated as primary ground truth. A future
lesion-area proxy, if built, is an auxiliary experiment only (validated
against this annotated set, never presented as equivalent to it).

## 1. Severity scale

Anchored to the Horsfall-Barratt convention (Horsfall & Barratt, 1945) --
grading by percentage of leaf area visibly affected -- rather than an
invented scale.

| Level | Label | Leaf-area-affected criterion |
|---|---|---|
| 0 | Healthy | No visible lesions/chlorosis/necrosis attributable to the diagnosed disease |
| 1 | Mild | <=10% of leaf area affected; discrete, non-coalescing lesions |
| 2 | Moderate | 10-40% affected; lesions beginning to coalesce |
| 3 | Severe | >40% affected; widespread coalesced lesions, major chlorosis/necrosis, possible defoliation-in-progress |

## 2. Visual criteria per level

- Healthy: uniform color/texture; any discoloration is clearly non-disease-pattern.
- Mild: countable discrete lesions, clear healthy tissue surrounding each, no merging.
- Moderate: lesions touching/merging in places; healthy tissue still the visual majority.
- Severe: healthy tissue a minority or absent; large merged affected regions; possible curling/wilting/partial death.

Ambiguity rules:
- Partially-visible leaf -> annotate visible portion only, flag `partial_leaf=1`.
- Apparent multi-disease co-occurrence -> exclude this round, flag `exclude_multi_disease=1`.
- Lighting/color-cast should be discounted via lesion texture/shape, not raw color intensity (annotator training point, not a hard rule).

## 3. Annotation guidelines

- Two independent annotators per image, blind to each other's labels and to the model's predicted disease label.
- Shared anchor set (2-3 example images per level) agreed before the main annotation pass.
- Session limit ~150 images before a break, to limit fatigue-driven drift.

## 4. Images per disease

Sample-size justification (proportion estimation, 95% CI, +-10% margin, worst-case p=0.5):
`n = 1.96^2 * 0.25 / 0.10^2 ~= 96` -> **~100 images per disease class**,
distributed across severity levels as the real data naturally allows (not
forced to uniform). For ~15-20 disease classes: **~1,500-2,000 images
total**. If resources are constrained, prioritize the disease classes
central to the paper's headline results and document reduced coverage
elsewhere as a limitation.

## 5. Stratification

Stratify the sampling pool by (source_dataset, crop, disease) -- the same
three fields already tracked per-Sample in Phase 2 -- using proportional-
within-disease random sampling, seeded with the project's existing seed
(42, from configs/config.yaml), with the sampled-image list saved to disk
BEFORE annotation begins.

## 6. Disagreement resolution

- Agreement -> accept as resolved_label.
- Off-by-one -> third adjudicator (ideally with plant-pathology background) assigns resolved_label; adjudicator identity recorded.
- Off-by-two-or-more -> treated as an ambiguous/borderline IMAGE, not an annotator error -- excluded from the labeled set with a logged reason. Never averaged/interpolated between annotators' labels -- that would synthesize a value neither annotator assigned.

## 7. Cohen's kappa computation

Ordinal scale -> use QUADRATIC-WEIGHTED kappa, not unweighted (unweighted
would penalize a Mild/Moderate disagreement identically to a
Healthy/Severe disagreement).

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotator_1_labels, annotator_2_labels, weights="quadratic")
```

Report overall kappa AND per-disease-class kappa (with the caveat that
small per-class samples, e.g. ~25 images per level per disease, will have
wide confidence intervals -- report the number, don't overstate precision).
Interpretation bands (Landis & Koch, 1977): <0.20 slight, 0.21-0.40 fair,
0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect -- state
alongside the actual computed number, never asserted alone.

## 8. CSV annotation format

```
image_filepath, source_dataset, crop_name, disease_name,
annotator_1_id, annotator_1_severity,
annotator_2_id, annotator_2_severity,
adjudicator_id, resolved_severity,
partial_leaf, exclude_multi_disease, exclusion_reason, notes
```

- `resolved_severity` is BLANK (not "-1", not "healthy") for any image not
  yet through resolution -- blank maps to "not annotated" in the pipeline
  (Section 9), never silently defaulted to a real class.
- `resolved_severity` values: `healthy`, `mild`, `moderate`, `severe`
  (strings, matching datasets/label_space.py's normalization convention).

## 9. Pipeline integration

See `datasets/severity_annotations.py` (loader + overlay + kappa
computation utility) and `datasets/unified.py`'s new optional
`severity_annotations_csv` parameter. Implementation notes:

- The overlay JOINS this CSV onto existing Phase-2 `Sample` objects by
  `filepath`, replacing `severity_name=None` with the real
  `resolved_severity` ONLY where a resolved (non-blank) label exists.
  Every other sample keeps `severity_name=None` -> encodes to
  `MISSING_LABEL=-1` exactly as already built in Phase 2 -- no change to
  that mechanism, no architecture change.
- Images with `resolved_severity` blank (still pending resolution) or
  `exclude_multi_disease=1` are treated as unannotated, not zero-filled.
- This CSV, once real annotation happens, is external, human-produced
  research data -- nothing in this codebase generates or infers its
  content. Until that CSV exists, `UnifiedCropDataset` behaves exactly as
  it did at the end of Phase 2 (all severity labels missing).

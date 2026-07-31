"""
datasets/severity_annotations.py

Loader + overlay + inter-rater agreement utility for the manually-created
severity annotation CSV described in research/severity_annotation_protocol.md
Section 8. No labels are generated or inferred here -- this module only
reads a CSV a human annotation process produces, and joins its
`resolved_severity` column onto existing Phase 2 `Sample` objects. Until
that CSV exists, every function here is inert / has nothing to load, and
UnifiedCropDataset behaves exactly as it did at the end of Phase 2.
"""

from __future__ import annotations

import csv
import dataclasses
import warnings
from pathlib import Path
from typing import Dict, List

from datasets.base_dataset import Sample

RESOLVED_SEVERITY_VALUES = {"healthy", "mild", "moderate", "severe"}
SEVERITY_ORDER = ["healthy", "mild", "moderate", "severe"]  # for ordinal kappa weighting

REQUIRED_COLUMNS = [
    "image_filepath", "source_dataset", "crop_name", "disease_name",
    "annotator_1_id", "annotator_1_severity",
    "annotator_2_id", "annotator_2_severity",
    "adjudicator_id", "resolved_severity",
    "partial_leaf", "exclude_multi_disease", "exclusion_reason", "notes",
]


@dataclasses.dataclass
class SeverityAnnotationStore:
    """In-memory view of the annotation CSV, keyed by exact filepath string
    (must match `str(Sample.filepath)` -- see load_severity_annotations'
    docstring for the practical caveat this implies)."""
    rows_by_filepath: Dict[str, dict]

    @property
    def num_rows(self) -> int:
        return len(self.rows_by_filepath)

    def resolved_count(self) -> int:
        return sum(
            1 for r in self.rows_by_filepath.values()
            if r.get("resolved_severity", "").strip() in RESOLVED_SEVERITY_VALUES
        )


def load_severity_annotations(csv_path) -> SeverityAnnotationStore:
    """Reads the annotation CSV. Raises a clear error if the header doesn't
    match the documented schema (research/severity_annotation_protocol.md
    Section 8), rather than silently reading a differently-shaped file and
    producing confusing downstream KeyErrors.

    CAVEAT, stated here rather than discovered later: the join key is the
    EXACT string `image_filepath` as written in the CSV, matched against
    `str(Sample.filepath)`. If the CSV is authored on one machine (absolute
    paths) and the dataset is later loaded from a different root on another
    machine, paths won't match. Recommendation: author the CSV with paths
    relative to each dataset's root and note the convention used -- this
    module does not currently normalize path formats itself, since doing so
    silently could mask a real join failure rather than surface it (see
    attach_severity_labels' mismatch-warning behavior for the analogous
    reasoning on crop/disease sanity-checking).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Severity annotation file not found: {csv_path}. This is "
            f"EXPECTED until manual annotation actually happens (see "
            f"research/severity_annotation_protocol.md) -- "
            f"UnifiedCropDataset works fine without it, just with "
            f"severity_name=None for every sample, same as end of Phase 2."
        )

    rows_by_filepath: Dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(
                f"{csv_path} is missing required column(s): {sorted(missing_cols)}. "
                f"Expected schema: {REQUIRED_COLUMNS} "
                f"(see research/severity_annotation_protocol.md Section 8)."
            )
        for row in reader:
            key = row["image_filepath"]
            if key in rows_by_filepath:
                raise ValueError(f"Duplicate image_filepath in {csv_path}: {key}")
            rows_by_filepath[key] = row

    return SeverityAnnotationStore(rows_by_filepath=rows_by_filepath)


def attach_severity_labels(samples: List[Sample], store: SeverityAnnotationStore) -> List[Sample]:
    """Returns a NEW list of Sample objects (Sample is a frozen dataclass,
    so this can't mutate in place) with severity_name filled in wherever
    `store` has a resolved (non-blank, non-excluded) label for that exact
    filepath. Every other sample keeps severity_name=None -- unchanged from
    Phase 2 behavior, still encoding to MISSING_LABEL(-1) downstream via
    the existing LabelSpace mechanism. No new masking/sentinel logic here.
    """
    out: List[Sample] = []
    mismatches: List[str] = []

    for sample in samples:
        key = str(sample.filepath)
        row = store.rows_by_filepath.get(key)

        if row is None:
            out.append(sample)
            continue

        if row.get("exclude_multi_disease") == "1":
            out.append(sample)  # excluded per protocol Section 2, stays unlabeled
            continue

        resolved = row.get("resolved_severity", "").strip()
        if resolved not in RESOLVED_SEVERITY_VALUES:
            out.append(sample)  # blank / not yet resolved -> stays unlabeled
            continue

        # Sanity cross-check: the CSV's own crop/disease columns should
        # agree with the Sample's. A mismatch means the filepath join
        # landed on the WRONG image (e.g. two datasets coincidentally share
        # a relative path) -- surfaced as a warning, label NOT attached,
        # rather than silently trusting a possibly-wrong join.
        if row.get("crop_name") != sample.crop_name or row.get("disease_name") != sample.disease_name:
            mismatches.append(key)
            out.append(sample)
            continue

        out.append(dataclasses.replace(sample, severity_name=resolved))

    if mismatches:
        warnings.warn(
            f"[severity_annotations] {len(mismatches)} row(s) matched a "
            f"filepath but had a crop/disease mismatch against the actual "
            f"Sample -- NOT attached, to avoid silently joining the wrong "
            f"annotation onto the wrong image. First few: {mismatches[:5]}"
        )

    return out


def compute_interrater_kappa(store: SeverityAnnotationStore, weights: str = "quadratic") -> dict:
    """Cohen's kappa over the RAW (pre-adjudication) annotator_1 vs
    annotator_2 labels, per protocol Section 7 -- computed on independent
    judgments, NEVER on resolved_severity (which would trivially inflate
    agreement, since resolved_severity is already the reconciled output)."""
    from sklearn.metrics import cohen_kappa_score

    order_index = {v: i for i, v in enumerate(SEVERITY_ORDER)}

    a1, a2 = [], []
    skipped = 0
    for row in store.rows_by_filepath.values():
        s1 = row.get("annotator_1_severity", "").strip()
        s2 = row.get("annotator_2_severity", "").strip()
        if s1 not in order_index or s2 not in order_index:
            skipped += 1
            continue
        a1.append(order_index[s1])
        a2.append(order_index[s2])

    if len(a1) < 2:
        raise ValueError(
            f"Need at least 2 fully-double-annotated rows to compute kappa, "
            f"found {len(a1)} (skipped {skipped} rows with an incomplete "
            f"or invalid annotator_1_severity/annotator_2_severity pair)."
        )

    kappa = cohen_kappa_score(a1, a2, weights=weights)
    return {
        "kappa": kappa,
        "weights": weights,
        "n_pairs": len(a1),
        "n_skipped_incomplete": skipped,
        "interpretation": interpret_kappa(kappa),
    }


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch, 1977 interpretation bands."""
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"

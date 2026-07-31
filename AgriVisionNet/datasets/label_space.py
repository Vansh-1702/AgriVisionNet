"""
datasets/label_space.py

Cross-dataset label unification. Each dataset loader (plantvillage.py etc.)
returns RAW STRING labels ("Apple", "Apple_scab"). This module builds a
single, consistent integer encoding shared across every enabled dataset --
e.g. "rice" must map to the same integer whether it came from
paddy_doctor.py or rice_disease.py.

WHY A SEPARATE MODULE, NOT DONE INSIDE EACH LOADER: if label encoding were
per-loader, adding a new dataset with an overlapping crop/disease name
(exactly the common case -- "rice" appears in two of our three datasets)
would require manually reconciling integer IDs across files. Fitting the
vocabulary once, over the union of all enabled datasets, is what makes
"add a new dataset with minimal code changes" actually true rather than
true-until-a-name-collides.

NORMALIZATION: string labels are lowercased and have separators normalized
(underscores/parentheses stripped to spaces, collapsed whitespace) before
being added to the vocabulary, so e.g. "Corn_(maize)" and "corn maize"
would collide into one entry rather than silently becoming two different
classes -- this is a real risk given our three datasets' documented
(and differently-formatted) naming conventions, not a hypothetical one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from datasets.base_dataset import MISSING_LABEL

PathLike = Union[str, Path]


def normalize_label(raw: str) -> str:
    """Canonicalize a raw string label for cross-dataset matching.

    "Corn_(maize)" -> "corn maize"
    "Apple_scab"    -> "apple scab"
    "rice"          -> "rice"
    """
    s = raw.lower()
    s = re.sub(r"[_\-()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class LabelSpace:
    """Bidirectional string<->int vocabulary for ONE label type (crop,
    disease, or severity). Built by calling `.fit(raw_labels)` once over
    every sample's raw label across every enabled dataset, then `.encode()`
    per-sample thereafter.
    """

    def __init__(self):
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}
        self._raw_display_name: Dict[str, str] = {}  # normalized -> a chosen display form
        self._fitted = False

    def fit(self, raw_labels: Iterable[str]) -> "LabelSpace":
        """Build the vocabulary. Deterministic: sorts normalized labels
        before assigning integer IDs, so re-fitting on the same underlying
        data always produces the same encoding (required for
        reproducibility -- an ID assignment that depended on iteration/
        dict order would silently invalidate saved checkpoints)."""
        normalized_to_display: Dict[str, str] = {}
        for raw in raw_labels:
            norm = normalize_label(raw)
            # keep the FIRST-seen display form for a given normalized label
            normalized_to_display.setdefault(norm, raw)

        sorted_norms = sorted(normalized_to_display.keys())
        self._str_to_int = {norm: i for i, norm in enumerate(sorted_norms)}
        self._int_to_str = {i: norm for norm, i in self._str_to_int.items()}
        self._raw_display_name = normalized_to_display
        self._fitted = True
        return self

    def encode(self, raw_label: Optional[str]) -> int:
        if raw_label is None:
            return MISSING_LABEL
        if not self._fitted:
            raise RuntimeError("LabelSpace.encode() called before .fit()")
        norm = normalize_label(raw_label)
        if norm not in self._str_to_int:
            raise KeyError(
                f"Label '{raw_label}' (normalized: '{norm}') was not seen "
                f"during .fit() -- this means a dataset was scanned AFTER "
                f"the label space was built/loaded, which would silently "
                f"produce wrong training labels. Refit (or load a saved "
                f"label space that actually covers this dataset) before "
                f"encoding."
            )
        return self._str_to_int[norm]

    def decode(self, idx: int) -> str:
        if idx == MISSING_LABEL:
            return "<missing>"
        norm = self._int_to_str[idx]
        return self._raw_display_name.get(norm, norm)

    @property
    def num_classes(self) -> int:
        return len(self._str_to_int)

    def save(self, path: PathLike) -> None:
        payload = {
            "str_to_int": self._str_to_int,
            "raw_display_name": self._raw_display_name,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: PathLike) -> "LabelSpace":
        payload = json.loads(Path(path).read_text())
        obj = cls()
        obj._str_to_int = {k: int(v) for k, v in payload["str_to_int"].items()}
        obj._int_to_str = {v: k for k, v in obj._str_to_int.items()}
        obj._raw_display_name = payload["raw_display_name"]
        obj._fitted = True
        return obj

    def __repr__(self) -> str:
        return f"LabelSpace(num_classes={self.num_classes}, fitted={self._fitted})"

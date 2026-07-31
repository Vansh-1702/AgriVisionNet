"""
datasets/integrity.py

Phase 2 objective 10: dataset integrity checks. These operate on a list of
`Sample` objects (or a concrete dataset's `.samples`), have no torch
dependency, and are fully exercised by this project's test suite in this
authoring sandbox.

Deliberately conservative about what counts as "corrupt": PIL's `.verify()`
invalidates the Image object for further use, so a corrupt-check re-opens
the file for verify() and discards that handle rather than reusing it --
a subtle bug if done wrong (using a post-verify() Image object for
anything else silently gives wrong pixel data in some PIL versions).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from PIL import Image, UnidentifiedImageError

from datasets.base_dataset import Sample


@dataclass
class IntegrityReport:
    total_samples: int
    corrupt_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    zero_byte_files: List[str] = field(default_factory=list)
    duplicate_groups: List[List[str]] = field(default_factory=list)  # each inner list = files with identical content hash
    class_counts: Dict[str, int] = field(default_factory=dict)       # keyed "crop|disease"
    imbalance_ratio: float = 0.0  # max class count / min class count
    is_clean: bool = True

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "num_corrupt": len(self.corrupt_files),
            "num_missing": len(self.missing_files),
            "num_zero_byte": len(self.zero_byte_files),
            "num_duplicate_groups": len(self.duplicate_groups),
            "num_duplicate_files": sum(len(g) for g in self.duplicate_groups),
            "num_classes": len(self.class_counts),
            "imbalance_ratio": self.imbalance_ratio,
            "is_clean": self.is_clean,
            "corrupt_files": self.corrupt_files,
            "missing_files": self.missing_files,
            "zero_byte_files": self.zero_byte_files,
            "duplicate_groups": self.duplicate_groups,
            "class_counts": self.class_counts,
        }


def _file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def run_integrity_checks(samples: List[Sample], check_duplicates: bool = True) -> IntegrityReport:
    """Run all Phase 2 objective-10 checks over a sample list.

    Args:
        samples: list of Sample (from a dataset's `.samples` after `_scan()`).
        check_duplicates: content-hash duplicate detection reads every file
            fully; can be disabled for a quick pass over very large datasets
            (the other checks are cheap: verify() reads headers/reduced data,
            not the full decoded image, for most formats).
    """
    report = IntegrityReport(total_samples=len(samples))
    class_counter: Counter = Counter()
    hash_to_files: Dict[str, List[str]] = defaultdict(list)

    for sample in samples:
        path = sample.filepath
        class_key = f"{sample.crop_name}|{sample.disease_name}"
        class_counter[class_key] += 1

        if not path.exists():
            report.missing_files.append(str(path))
            continue

        size = path.stat().st_size
        if size == 0:
            report.zero_byte_files.append(str(path))
            continue

        try:
            with Image.open(path) as img:
                img.verify()  # raises on truncated/corrupt data; invalidates `img` for further use
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            report.corrupt_files.append(f"{path} ({type(e).__name__}: {e})")
            continue

        if check_duplicates:
            try:
                h = _file_hash(path)
                hash_to_files[h].append(str(path))
            except OSError as e:
                # file existed and passed verify() moments ago but became
                # unreadable (race condition / permissions) -- report, don't crash
                report.corrupt_files.append(f"{path} (hash read failed: {e})")

    if check_duplicates:
        report.duplicate_groups = [files for files in hash_to_files.values() if len(files) > 1]

    report.class_counts = dict(class_counter)
    if class_counter:
        counts = list(class_counter.values())
        report.imbalance_ratio = max(counts) / max(min(counts), 1)

    report.is_clean = (
        len(report.corrupt_files) == 0
        and len(report.missing_files) == 0
        and len(report.zero_byte_files) == 0
        and len(report.duplicate_groups) == 0
    )
    return report


# Imbalance-ratio interpretation thresholds -- not enforced automatically
# (a real agricultural dataset is very likely imbalanced; failing the build
# over it would be the wrong default), but exposed so training/eval scripts
# can log a clear warning at the right severity.
IMBALANCE_MILD = 3.0
IMBALANCE_SEVERE = 10.0


def imbalance_severity(ratio: float) -> str:
    if ratio >= IMBALANCE_SEVERE:
        return "severe"
    if ratio >= IMBALANCE_MILD:
        return "mild"
    return "none"

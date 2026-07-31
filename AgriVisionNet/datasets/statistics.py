"""
datasets/statistics.py

Phase 2 objective 4: automatic dataset statistics. Computes per-class
counts, image-size distribution, and per-dataset provenance breakdown from
a sample list. No torch dependency -- exercised directly in this sandbox.

Deliberately does NOT compute per-channel mean/std for normalization here,
even though that's a common "dataset statistics" ask -- that requires
decoding every image's full pixel data (expensive for large datasets) and
belongs in a separate, explicitly-invoked script
(`datasets/compute_normalization_stats.py`, Phase 3, since normalization
constants are a training-time concern, not a Phase 2 validation concern) --
noted here so it isn't assumed to be silently missing by omission.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from datasets.base_dataset import Sample


@dataclass
class DatasetStatistics:
    total_samples: int
    per_source_counts: Dict[str, int] = field(default_factory=dict)
    per_crop_counts: Dict[str, int] = field(default_factory=dict)
    per_disease_counts: Dict[str, int] = field(default_factory=dict)
    per_crop_disease_counts: Dict[str, int] = field(default_factory=dict)  # "Crop|Disease" -> count
    severity_coverage: Dict[str, int] = field(default_factory=dict)  # {"labeled": n, "missing": n}
    image_size_distribution: Dict[str, int] = field(default_factory=dict)  # "WxH" -> count
    num_unique_image_sizes: int = 0

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "per_source_counts": self.per_source_counts,
            "per_crop_counts": self.per_crop_counts,
            "per_disease_counts": self.per_disease_counts,
            "per_crop_disease_counts": self.per_crop_disease_counts,
            "severity_coverage": self.severity_coverage,
            "num_unique_image_sizes": self.num_unique_image_sizes,
            "image_size_distribution": self.image_size_distribution,
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def compute_dataset_statistics(samples: List[Sample], probe_image_sizes: bool = True) -> DatasetStatistics:
    """
    Args:
        samples: list of Sample.
        probe_image_sizes: if True, opens each image (header-only via PIL,
            does not decode full pixel data for most formats) to record its
            (width, height). Can be disabled for a fast pass that skips
            filesystem access entirely (source/crop/disease/severity counts
            only need the in-memory Sample metadata).
    """
    stats = DatasetStatistics(total_samples=len(samples))

    source_counter: Counter = Counter()
    crop_counter: Counter = Counter()
    disease_counter: Counter = Counter()
    crop_disease_counter: Counter = Counter()
    severity_labeled = 0
    severity_missing = 0
    size_counter: Counter = Counter()

    for sample in samples:
        source_counter[sample.source_dataset] += 1
        crop_counter[sample.crop_name] += 1
        disease_counter[sample.disease_name] += 1
        crop_disease_counter[f"{sample.crop_name}|{sample.disease_name}"] += 1

        if sample.severity_name is not None:
            severity_labeled += 1
        else:
            severity_missing += 1

        if probe_image_sizes and sample.filepath.exists():
            try:
                with Image.open(sample.filepath) as img:
                    size_counter[f"{img.width}x{img.height}"] += 1
            except Exception:
                # a genuinely corrupt/unreadable file here is integrity.py's
                # job to report in detail; statistics.py just skips it
                # rather than duplicating that error-reporting logic.
                pass

    stats.per_source_counts = dict(source_counter)
    stats.per_crop_counts = dict(crop_counter)
    stats.per_disease_counts = dict(disease_counter)
    stats.per_crop_disease_counts = dict(crop_disease_counter)
    stats.severity_coverage = {"labeled": severity_labeled, "missing": severity_missing}
    stats.image_size_distribution = dict(size_counter)
    stats.num_unique_image_sizes = len(size_counter)

    return stats

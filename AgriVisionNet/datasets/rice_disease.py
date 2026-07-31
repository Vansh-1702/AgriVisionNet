"""
datasets/rice_disease.py

Loader for a "Rice Disease" image dataset.

============================================================================
IMPORTANT AMBIGUITY -- FLAGGED, NOT RESOLVED
============================================================================
"Rice Disease Dataset" is NOT a single canonical dataset name -- at least
two distinct, similarly-named public datasets exist (e.g. a Mendeley-hosted
"Rice Diseases Image Dataset" with train/validation folders, and various
Kaggle uploads with different class sets and folder layouts). Which one you
mean was never confirmed in this project's discussion. This loader is
written against ONE assumed, commonly-seen layout below; if it doesn't
match your actual download, that is expected and should be treated as
"this loader needs a one-line path adjustment," not "this loader is
broken."

============================================================================
ASSUMED DIRECTORY STRUCTURE -- NOT YET VERIFIED
============================================================================
Two layouts are supported, tried in this order:

  Layout A (split subfolders present):
    <root>/train/<ClassName>/*.jpg
    <root>/validation/<ClassName>/*.jpg   (or "val")

  Layout B (no split subfolders -- class folders directly under root):
    <root>/<ClassName>/*.jpg

If Layout A is found, the split membership from the source dataset is
PRESERVED as `source_split` metadata (not silently discarded) even though
Phase 2's own train/val/test splitting (datasets/splitting.py) will
re-split everything from scratch for experimental consistency across all
three datasets -- see datasets/README.md for why we don't just use each
dataset's own bundled split.

Crop is fixed to "rice" for every sample -- same reasoning as
paddy_doctor.py.

============================================================================
KNOWN DATA GAP -- SEVERITY
============================================================================
No severity annotation is assumed present in either candidate public
dataset. All samples: severity_name=None -> MISSING_LABEL(-1).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from datasets.base_dataset import BaseCropDataset, Sample
from datasets.registry import register_dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SPLIT_DIR_CANDIDATES = {"train": "train", "validation": "val", "val": "val", "test": "test"}


@register_dataset("rice_disease")
class RiceDiseaseDataset(BaseCropDataset):
    """See module docstring -- layout is genuinely ambiguous, see the flag above."""

    def _scan(self) -> List[Sample]:
        split_dirs = [d for d in self.root.iterdir() if d.is_dir() and d.name.lower() in SPLIT_DIR_CANDIDATES]

        if split_dirs:
            return self._scan_layout_a(split_dirs)
        return self._scan_layout_b()

    def _scan_layout_a(self, split_dirs: List[Path]) -> List[Sample]:
        samples: List[Sample] = []
        for split_dir in sorted(split_dirs):
            class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
            for class_dir in class_dirs:
                disease = class_dir.name
                for f in sorted(class_dir.iterdir()):
                    if f.suffix in IMAGE_EXTENSIONS:
                        samples.append(
                            Sample(
                                filepath=f,
                                crop_name="rice",
                                disease_name="healthy" if disease.lower() in ("healthy", "normal") else disease,
                                source_dataset="rice_disease",
                                severity_name=None,
                            )
                        )
        return samples

    def _scan_layout_b(self) -> List[Sample]:
        samples: List[Sample] = []
        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())

        if len(class_dirs) == 0:
            raise ValueError(
                f"[rice_disease] {self.root} has neither split subfolders "
                f"(train/validation) nor class subfolders directly under "
                f"root. This dataset's actual layout doesn't match either "
                f"assumed layout in this module's docstring -- the ambiguity "
                f"flagged there is real; inspect your download and adjust "
                f"this loader rather than assuming the code is wrong."
            )

        for class_dir in class_dirs:
            disease = class_dir.name
            for f in sorted(class_dir.iterdir()):
                if f.suffix in IMAGE_EXTENSIONS:
                    samples.append(
                        Sample(
                            filepath=f,
                            crop_name="rice",
                            disease_name="healthy" if disease.lower() in ("healthy", "normal") else disease,
                            source_dataset="rice_disease",
                            severity_name=None,
                        )
                    )
        return samples

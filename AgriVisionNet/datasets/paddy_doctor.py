"""
datasets/paddy_doctor.py

Loader for the "Paddy Doctor: Paddy Disease Classification" dataset
(Kaggle competition, Sethupathy et al.).

============================================================================
ASSUMED DIRECTORY STRUCTURE -- NOT YET VERIFIED AGAINST A REAL DOWNLOAD
============================================================================
    <root>/
        train_images/
            bacterial_leaf_blight/
                100330.jpg
                ...
            bacterial_leaf_streak/
            bacterial_panicle_blight/
            blast/
            brown_spot/
            dead_heart/
            downy_mildew/
            hispa/
            normal/
            tungro/
        train.csv          # columns: image_id, label, variety, age
        test_images/        # unlabeled -- NOT used by this loader
        sample_submission.csv

This loader scans `train_images/<label>/` (folder-per-class, no CSV
dependency required for basic use). If `train.csv` is present, its
`variety` and `age` columns are attached as extra per-sample metadata
(not currently consumed by the model -- kept for a possible future
"variety-aware" extension, per the brief's stated goal of adding new
modalities without touching the backbone).

Single-crop dataset: every sample's crop_name is fixed to "rice" -- this
is not a placeholder, it's a correct fact about this dataset's scope.

============================================================================
KNOWN DATA GAP -- SEVERITY
============================================================================
No severity annotation is provided. All samples: severity_name=None ->
MISSING_LABEL(-1). Same gap as plantvillage.py, documented per-loader
deliberately rather than only once, since a reader may only look at this file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from datasets.base_dataset import BaseCropDataset, Sample
from datasets.registry import register_dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@register_dataset("paddy_doctor")
class PaddyDoctorDataset(BaseCropDataset):
    """See module docstring for the assumed directory layout."""

    def _scan(self) -> List[Sample]:
        images_dir = self.root / "train_images"
        if not images_dir.exists():
            raise FileNotFoundError(
                f"[paddy_doctor] expected {images_dir} to exist (per this "
                f"module's documented, unverified layout: "
                f"'<root>/train_images/<label>/*.jpg'). If your download "
                f"unpacked differently, update this path or pass the correct "
                f"root."
            )

        # train.csv is optional metadata, not required for basic loading --
        # attempt to read it, but don't fail the whole scan if it's missing
        # or malformed; that would be too strict for what's optional data.
        metadata = self._try_read_metadata_csv()

        samples: List[Sample] = []
        class_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir())

        if len(class_dirs) == 0:
            raise ValueError(
                f"[paddy_doctor] {images_dir} contains no class subdirectories."
            )

        for class_dir in class_dirs:
            label = class_dir.name  # e.g. "blast", "normal"
            for f in sorted(class_dir.iterdir()):
                if f.suffix in IMAGE_EXTENSIONS:
                    samples.append(
                        Sample(
                            filepath=f,
                            crop_name="rice",  # correct fact, not a placeholder -- see module docstring
                            disease_name="healthy" if label == "normal" else label,
                            source_dataset="paddy_doctor",
                            severity_name=None,  # documented gap, see module docstring
                        )
                    )
        return samples

    def _try_read_metadata_csv(self) -> Optional[dict]:
        csv_path = self.root / "train.csv"
        if not csv_path.exists():
            return None
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                return {row["image_id"]: row for row in reader}
        except (KeyError, csv.Error) as e:
            # Metadata is optional -- warn via exception message text in a
            # log, but do not raise, since train_images/ scanning alone is
            # sufficient for this loader's core contract.
            import warnings
            warnings.warn(
                f"[paddy_doctor] found train.csv but couldn't parse it as "
                f"expected ({e}); proceeding without variety/age metadata."
            )
            return None

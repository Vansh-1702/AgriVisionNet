"""
datasets/base_dataset.py

Unified dataset interface for AgriVisionNet.

DESIGN GOAL (Phase 2 objective 8 — "support adding new datasets with minimal
code changes"): every concrete dataset loader (plantvillage.py,
paddy_doctor.py, rice_disease.py, and any future dataset) implements exactly
ONE method — `_scan()` -> list[Sample]. Image loading, the __getitem__
contract, and integration with statistics/integrity/splitting are handled
once, here. Adding dataset N+1 means: write a `_scan()` method, register it
(datasets/registry.py), add one block to configs/config.yaml. No other file
needs to change.

TORCH IS AN OPTIONAL IMPORT in this module specifically. The path-scanning
logic has no tensor dependency, so it is exercised directly by the Phase 2
test suite in this authoring sandbox, which has no torch/GPU install (see
research/reproducibility.md). `__getitem__`'s tensor output obviously
requires torch at actual training time — that path is written correctly but
UNVERIFIED here, same caveat as Phase 1.

LABEL ENCODING IS DELIBERATELY NOT DONE HERE. This class returns raw string
labels ("Apple", "Apple_scab", None for missing severity). Turning strings
into the model's integer label space is a cross-dataset concern (the same
"Tomato" must map to the same integer whether it came from PlantVillage or
a future added dataset) and lives in datasets/label_space.py +
datasets/encoded_dataset.py instead, keeping this class dataset-structure-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised in this authoring sandbox
    _HAS_TORCH = False

    class _TorchDataset:  # minimal stand-in so class definitions don't fail
        """Fallback used only when torch isn't installed (this sandbox).
        Real training environments must have torch; this only exists so
        the non-tensor parts of this module (scanning, statistics,
        integrity, splitting) can be imported and tested without it."""
        pass


# Sentinel for "this dataset doesn't provide this label" -- MUST match
# losses/evidential_loss.py's `ignore_index=-1` default so a missing label
# is transparently masked out of that task's loss with no extra glue code.
MISSING_LABEL = -1


@dataclass(frozen=True)
class Sample:
    """One dataset record, BEFORE label-space encoding (raw string labels).

    Attributes:
        filepath: path to the image file.
        crop_name: raw crop name as it appears in the source dataset
            (e.g. "Apple", "rice"). Never None -- every one of our three
            documented source datasets identifies the crop, even if only
            implicitly (Paddy Doctor / Rice Disease are single-crop: rice).
        disease_name: raw disease name (e.g. "Apple_scab", "healthy", "blast").
        source_dataset: name of the dataset this sample came from (kept for
            provenance in statistics reports and for debugging label
            collisions across datasets).
        severity_name: raw severity label, or None if the source dataset
            doesn't provide one. IMPORTANT, flagged prominently rather than
            silently handled: none of the three datasets documented in
            plantvillage.py / paddy_doctor.py / rice_disease.py provide
            severity ground truth. Every sample from all three currently
            has severity_name=None -> encodes to MISSING_LABEL. This is a
            real, unresolved data gap for the severity task, not an
            implementation placeholder -- see datasets/README.md.
    """
    filepath: Path
    crop_name: str
    disease_name: str
    source_dataset: str
    severity_name: Optional[str] = None


class BaseCropDataset(_TorchDataset):
    """Abstract base class. Concrete subclasses implement `_scan()` only.

    Subclass contract:
        - set `name: str` (used in error messages, statistics, provenance)
        - implement `_scan(self) -> List[Sample]`
        - do NOT override __init__, __len__, or __getitem__ unless a
          dataset genuinely needs a different image-loading strategy
          (e.g. a dataset shipped as a single HDF5/LMDB file rather than
          loose image files) -- if so, override `_load_image(self, sample)`
          instead, which is the one seam designed for that.
    """

    name: str = "base"

    def __init__(self, root: Union[str, Path], transform: Optional[Callable] = None):
        """
        Args:
            root: dataset root directory. Must already exist -- this class
                deliberately does NOT create directories or download data;
                that would silently mask a misconfigured path.
            transform: an Albumentations `Compose` pipeline (callable with
                an `image=np.ndarray` kwarg, returning a dict with an
                "image" key) as built by datasets/transforms.py, or None
                (raw PIL image passed through, useful for statistics/
                integrity scans that don't need augmented tensors).
        """
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"[{self.name}] dataset root does not exist: {self.root}\n"
                f"See datasets/{self.name}.py's module docstring for the "
                f"expected directory layout (documented from public-source "
                f"knowledge, flagged as an assumption to verify against your "
                f"actual download -- see datasets/README.md)."
            )

        self.samples: List[Sample] = self._scan()

        if len(self.samples) == 0:
            raise ValueError(
                f"[{self.name}] scanned {self.root} but found zero valid "
                f"samples. Either the directory layout doesn't match what "
                f"{self.name}.py expects (check datasets/README.md's "
                f"documented-assumption section), or the directory is empty."
            )

    def _scan(self) -> List[Sample]:
        """Must be implemented by every concrete dataset loader."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _scan()")

    def _load_image(self, sample: Sample) -> Image.Image:
        """Seam for datasets with a non-loose-file storage format to override.
        Default: load a standalone image file from disk."""
        return Image.open(sample.filepath).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if not _HAS_TORCH:
            raise RuntimeError(
                "torch is required for __getitem__ (tensor image output) but "
                "is not installed in this environment. The scanning/"
                "statistics/integrity methods on this class work fine "
                "without torch (that's what this sandbox's tests exercise); "
                "actual image tensor loading does not."
            )

        sample = self.samples[idx]
        pil_image = self._load_image(sample)

        if self.transform is not None:
            np_image = np.array(pil_image)
            augmented = self.transform(image=np_image)
            image_out = augmented["image"]  # already a torch tensor per Albumentations' ToTensorV2
        else:
            image_out = pil_image  # caller wanted raw PIL (e.g. statistics scan)

        return {
            "image": image_out,
            "filepath": str(sample.filepath),
            "crop_name": sample.crop_name,
            "disease_name": sample.disease_name,
            "severity_name": sample.severity_name,
            "source_dataset": sample.source_dataset,
        }

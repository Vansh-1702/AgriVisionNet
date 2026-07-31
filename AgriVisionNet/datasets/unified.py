"""
datasets/unified.py

Combines every ENABLED dataset from configs/config.yaml's `data.datasets`
list into one PyTorch-ready Dataset with a shared crop/disease/severity
label space (datasets/label_space.py), built once over the union of all
enabled datasets' raw samples.

This is the object training/eval scripts (Phase 3) actually import --
individual loaders (plantvillage.py etc.) are not meant to be used
directly for training, only through this wrapper, so label encoding is
never accidentally skipped or duplicated per-dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import datasets  # noqa: F401 -- import side-effect populates the registry
from datasets.base_dataset import BaseCropDataset, MISSING_LABEL, Sample
from datasets.label_space import LabelSpace
from datasets.registry import build_dataset
from datasets.severity_annotations import load_severity_annotations, attach_severity_labels

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - this authoring sandbox
    _HAS_TORCH = False

    class _TorchDataset:
        pass


class UnifiedCropDataset(_TorchDataset):
    """
    Args:
        dataset_configs: list of dicts like
            [{"name": "plantvillage", "root": "data/plantvillage/", "enabled": True}, ...]
            -- exactly the shape of configs/config.yaml's `data.datasets`.
        transform: Albumentations pipeline (see datasets/transforms.py),
            applied identically to every underlying dataset's images.
        label_spaces: optionally pass in already-fitted LabelSpace objects
            (crop/disease/severity) -- e.g. the ones fitted on a train
            split, reused UNCHANGED for val/test splits so val/test never
            see a label space that includes classes absent from train
            (which would be encoding leakage, not a training benefit).
            If None, fits fresh label spaces from THIS wrapper's own data
            (only correct to do this for a train-split construction).
        severity_annotations_csv: optional path to the manually-created
            severity annotation CSV (research/severity_annotation_protocol.md
            Section 8). If given, resolved severity labels are joined onto
            the raw samples BEFORE label-space fitting/encoding, so the
            severity label space reflects whatever real annotations exist.
            If None (default) or the file doesn't exist yet, every sample's
            severity_name stays None, exactly as at the end of Phase 2 --
            no behavior change for anyone not yet using annotations.
    """

    def __init__(
        self,
        dataset_configs: List[dict],
        transform=None,
        label_spaces: Optional[Dict[str, LabelSpace]] = None,
        severity_annotations_csv: Optional[str] = None,
    ):
        self.transform = transform
        self._sub_datasets: List[BaseCropDataset] = []
        self._sample_index: List[tuple] = []  # (sub_dataset_idx, local_idx) -- kept for get_raw_sample's provenance lookup, NOT used to fetch samples for __getitem__ (see below)

        enabled_configs = [c for c in dataset_configs if c.get("enabled", False)]
        if not enabled_configs:
            raise ValueError(
                "No dataset in dataset_configs has enabled=True. Check "
                "configs/config.yaml's `data.datasets` list -- at least one "
                "entry must be enabled to build a UnifiedCropDataset."
            )

        for entry in enabled_configs:
            ds = build_dataset(name=entry["name"], root=entry["root"], transform=None)
            self._sub_datasets.append(ds)
            for local_idx in range(len(ds)):
                self._sample_index.append((len(self._sub_datasets) - 1, local_idx))

        all_samples: List[Sample] = [
            s for ds in self._sub_datasets for s in ds.samples
        ]

        if severity_annotations_csv is not None and Path(severity_annotations_csv).exists():
            store = load_severity_annotations(severity_annotations_csv)
            all_samples = attach_severity_labels(all_samples, store)
        # NOTE: if severity_annotations_csv is given but the file doesn't
        # exist yet, this is silently treated as "no annotations yet" --
        # deliberate, since research/severity_annotation_protocol.md's
        # annotation process hasn't produced the file yet at time of
        # writing, and a training/data-pipeline smoke-test run shouldn't
        # hard-fail just because annotation is still in progress. A caller
        # that wants to REQUIRE the file's presence should check for it
        # explicitly before constructing this class.

        # This list (POST severity overlay, if any) is the single source of
        # truth for both label-space fitting below AND __getitem__/
        # get_raw_sample -- an earlier version of this class read directly
        # from `self._sub_datasets[i].samples` in those two methods, which
        # bypassed the overlay entirely (the overlay produces NEW Sample
        # objects via dataclasses.replace, since Sample is frozen; the
        # original sub-dataset's list is never mutated). Fixed by making
        # _all_samples the only place __getitem__/get_raw_sample read from.
        self._all_samples = all_samples

        if label_spaces is not None:
            self.crop_space = label_spaces["crop"]
            self.disease_space = label_spaces["disease"]
            self.severity_space = label_spaces["severity"]
        else:
            self.crop_space = LabelSpace().fit(s.crop_name for s in all_samples)
            self.disease_space = LabelSpace().fit(s.disease_name for s in all_samples)
            self.severity_space = LabelSpace().fit(
                s.severity_name for s in all_samples if s.severity_name is not None
            )

    def label_spaces(self) -> Dict[str, LabelSpace]:
        return {"crop": self.crop_space, "disease": self.disease_space, "severity": self.severity_space}

    def __len__(self) -> int:
        return len(self._all_samples)

    def get_raw_sample(self, idx: int) -> Sample:
        return self._all_samples[idx]

    def __getitem__(self, idx: int) -> dict:
        sample = self._all_samples[idx]
        sub_idx, _ = self._sample_index[idx]
        sub_ds = self._sub_datasets[sub_idx]  # only used for its _load_image method

        if not _HAS_TORCH:
            raise RuntimeError(
                "torch is required for __getitem__ but is not installed in "
                "this environment (see README.md's sandbox-limitation note)."
            )

        pil_image = sub_ds._load_image(sample)

        if self.transform is not None:
            import numpy as np
            np_image = np.array(pil_image)
            image_out = self.transform(image=np_image)["image"]
        else:
            image_out = pil_image

        return {
            "image": image_out,
            "crop_label": self.crop_space.encode(sample.crop_name),
            "disease_label": self.disease_space.encode(sample.disease_name),
            "severity_label": (
                self.severity_space.encode(sample.severity_name)
                if sample.severity_name is not None
                else MISSING_LABEL
            ),
            "source_dataset": sample.source_dataset,
            "filepath": str(sample.filepath),
        }

    def all_samples(self) -> List[Sample]:
        """Exposes the flattened raw sample list (post severity-overlay, if
        any) -- used by datasets/splitting.py, datasets/statistics.py, and
        datasets/integrity.py, all of which operate on List[Sample], not
        on this wrapper directly."""
        return self._all_samples

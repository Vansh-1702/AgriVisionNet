"""
datasets/__init__.py

Importing this package registers every built-in dataset loader (registry
population happens via the @register_dataset decorator as an import
side-effect). Any code that calls datasets.registry.build_dataset(...)
must import `datasets` first (or import the specific loader module) --
this is standard for registry patterns but easy to forget, hence this
explicit, commented import list rather than relying on it happening
implicitly.

Adding a new dataset (Phase 2 objective 8): write datasets/your_dataset.py
with a @register_dataset("your_name") class, then add one import line here.
"""

from datasets.plantvillage import PlantVillageDataset   # noqa: F401
from datasets.paddy_doctor import PaddyDoctorDataset     # noqa: F401
from datasets.rice_disease import RiceDiseaseDataset     # noqa: F401

__all__ = ["PlantVillageDataset", "PaddyDoctorDataset", "RiceDiseaseDataset"]

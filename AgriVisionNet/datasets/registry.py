"""
datasets/registry.py

A name -> class registry so `configs/config.yaml`'s `data.datasets[*].name`
string is the ONLY place that selects which loader runs. This is what makes
"add a new dataset with minimal code changes" concrete: write the loader
class, decorate it with `@register_dataset("your_name")`, add a config
block -- nothing else in the codebase needs to know it exists.
"""

from __future__ import annotations

from typing import Dict, Type

from datasets.base_dataset import BaseCropDataset

_REGISTRY: Dict[str, Type[BaseCropDataset]] = {}


def register_dataset(name: str):
    """Class decorator. Usage:

        @register_dataset("plantvillage")
        class PlantVillageDataset(BaseCropDataset):
            ...
    """
    def _decorator(cls: Type[BaseCropDataset]) -> Type[BaseCropDataset]:
        if name in _REGISTRY:
            raise ValueError(
                f"Dataset name '{name}' is already registered to "
                f"{_REGISTRY[name].__name__}. Names must be unique -- pick "
                f"a different name or check for an accidental duplicate "
                f"import."
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _decorator


def build_dataset(name: str, root: str, transform=None) -> BaseCropDataset:
    """Factory used by training/eval scripts and datasets/unified.py.

    Raises a clear error listing available names if `name` isn't registered,
    rather than a bare KeyError.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none registered yet)"
        raise KeyError(
            f"No dataset registered under name '{name}'. Available: {available}. "
            f"If you're adding a new dataset, confirm its module is imported "
            f"somewhere before build_dataset() is called (import side-effects "
            f"are what populate this registry) -- see datasets/__init__.py."
        )
    return _REGISTRY[name](root=root, transform=transform)


def available_datasets() -> list:
    return sorted(_REGISTRY.keys())

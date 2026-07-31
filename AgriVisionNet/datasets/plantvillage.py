"""
datasets/plantvillage.py

Loader for the PlantVillage dataset (Hughes & Salathé, 2015; the commonly
redistributed Kaggle mirrors use the same folder convention).

============================================================================
ASSUMED DIRECTORY STRUCTURE -- NOT YET VERIFIED AGAINST A REAL DOWNLOAD
============================================================================
This structure is documented from public-source knowledge of how
PlantVillage is conventionally packaged (both the original dataset and the
common "New Plant Diseases Dataset (Augmented)" Kaggle mirror). It has NOT
been checked against an actual downloaded copy in this project -- confirm
against yours before trusting this loader's output. If your copy differs,
see datasets/README.md for what to change.

    <root>/
        Apple___Apple_scab/
            0a5....JPG
            ...
        Apple___Black_rot/
        Apple___Cedar_apple_rust/
        Apple___healthy/
        Corn_(maize)___Common_rust_/
        Corn_(maize)___healthy/
        Tomato___Tomato_Yellow_Leaf_Curl_Virus/
        Tomato___healthy/
        ... (38 folders total in the standard release)

Folder-name convention: "{Crop}___{Disease}" -- a TRIPLE underscore
separates crop from disease (confirmed convention in the standard release;
verify your copy uses the same separator and not a different one, e.g. a
single underscore or a space, before relying on the parse below).

============================================================================
KNOWN DATA GAP -- SEVERITY
============================================================================
PlantVillage, as documented, provides NO severity annotation. Every sample
from this loader has severity_name=None -> encodes to MISSING_LABEL(-1).
This is a real, unresolved gap for the severity task (flagged in
research/design_rationale.md and datasets/base_dataset.py's Sample
docstring too, deliberately repeated here since this is the loader most
likely to be a project's primary dataset) -- not something to quietly work
around by inventing labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from datasets.base_dataset import BaseCropDataset, Sample
from datasets.registry import register_dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_plantvillage_folder_name(folder_name: str) -> tuple:
    """Split a PlantVillage class-folder name into (crop, disease).

    Raises ValueError on a folder name that doesn't match the documented
    "{Crop}___{Disease}" convention, rather than silently guessing -- a
    silent fallback here would be far more dangerous than a loud failure,
    since it would mislabel real training data.
    """
    if "___" not in folder_name:
        raise ValueError(
            f"Folder name '{folder_name}' does not contain the expected "
            f"'___' (triple underscore) crop/disease separator. Either this "
            f"isn't a PlantVillage-formatted directory, or your copy uses a "
            f"different naming convention than documented in this module's "
            f"docstring -- update parse_plantvillage_folder_name() to match "
            f"your actual data rather than silently skipping the folder."
        )
    crop, disease = folder_name.split("___", maxsplit=1)
    return crop.strip(), disease.strip()


@register_dataset("plantvillage")
class PlantVillageDataset(BaseCropDataset):
    """See module docstring for the assumed directory layout."""

    def _scan(self) -> List[Sample]:
        samples: List[Sample] = []
        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())

        if len(class_dirs) == 0:
            raise ValueError(
                f"[plantvillage] {self.root} contains no subdirectories. "
                f"Expected one subdirectory per '{{Crop}}___{{Disease}}' "
                f"class, per this module's documented (unverified) layout."
            )

        for class_dir in class_dirs:
            crop, disease = parse_plantvillage_folder_name(class_dir.name)
            for f in sorted(class_dir.iterdir()):
                if f.suffix in IMAGE_EXTENSIONS:
                    samples.append(
                        Sample(
                            filepath=f,
                            crop_name=crop,
                            disease_name=disease,
                            source_dataset="plantvillage",
                            severity_name=None,  # documented gap, see module docstring
                        )
                    )
        return samples

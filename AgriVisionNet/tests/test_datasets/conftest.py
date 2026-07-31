"""
tests/test_datasets/conftest.py

Shared pytest fixtures. All image data here is SYNTHETIC (tiny solid-color
PIL images), used ONLY to test loader/scanning/integrity logic -- never
used for training or reported as experimental data, per the user's explicit
instruction. Content uniqueness matters for the duplicate-detection tests,
so a global counter is used to guarantee distinct pixel values across every
generated image (an earlier hand-run version of this exact mistake --
reusing a per-class loop index for color -- produced accidental cross-class
duplicate images and was caught by the integrity checker itself; fixed here
by using a single monotonic counter across all fixture-creation helpers).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from PIL import Image

_color_counter = itertools.count(0)


def _make_image(path: Path, size=(16, 16)):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = next(_color_counter) * 7 % 256
    Image.new("RGB", size, color=(c, (c * 3) % 256, (c * 5) % 256)).save(path)


@pytest.fixture
def plantvillage_root(tmp_path) -> Path:
    root = tmp_path / "plantvillage"
    layout = {
        "Apple___Apple_scab": 3,
        "Apple___healthy": 2,
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 2,
        "Corn_(maize)___Common_rust_": 2,
    }
    for cls, n in layout.items():
        for i in range(n):
            _make_image(root / cls / f"img{i}.jpg")
    return root


@pytest.fixture
def paddy_doctor_root(tmp_path) -> Path:
    root = tmp_path / "paddy_doctor"
    images_dir = root / "train_images"
    for cls, n in {"blast": 2, "normal": 2, "brown_spot": 2}.items():
        for i in range(n):
            _make_image(images_dir / cls / f"p{i}.jpg")

    csv_path = root / "train.csv"
    csv_path.write_text(
        "image_id,label,variety,age\n"
        "p0.jpg,blast,ADT45,45\n"
        "p1.jpg,blast,ADT45,50\n"
    )
    return root


@pytest.fixture
def rice_disease_root_layout_a(tmp_path) -> Path:
    """Layout A: split subfolders (train/validation) present."""
    root = tmp_path / "rice_disease_a"
    for split in ["train", "validation"]:
        for cls, n in {"BrownSpot": 2, "Healthy": 2}.items():
            for i in range(n):
                _make_image(root / split / cls / f"r{i}.jpg")
    return root


@pytest.fixture
def rice_disease_root_layout_b(tmp_path) -> Path:
    """Layout B: class folders directly under root, no split subfolders."""
    root = tmp_path / "rice_disease_b"
    for cls, n in {"BrownSpot": 2, "Healthy": 2}.items():
        for i in range(n):
            _make_image(root / cls / f"r{i}.jpg")
    return root

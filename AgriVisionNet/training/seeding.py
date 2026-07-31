"""
training/seeding.py

Implements exactly the seeding protocol already committed to in
research/reproducibility.md Section 2 -- this module doesn't introduce a
new policy, it implements the one already documented.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seeds every RNG source this project's data pipeline (Phase 2) and
    training loop (Phase 3) actually touch. torch seeding is included but
    guarded -- this function must be safely callable in this torch-free
    authoring sandbox too (e.g. from a script validating Phase 2 logic).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # expected in this authoring sandbox; real training envs must have torch


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(..., worker_init_fn=seed_worker) -- without this,
    each dataloader worker process gets its own unseeded numpy/random state
    even if the main process was seeded, per research/reproducibility.md
    Section 2's explicit warning about this exact gap."""
    worker_seed = (torch_initial_seed_or_fallback() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_initial_seed_or_fallback(fallback: int = 42) -> int:
    try:
        import torch
        return torch.initial_seed() % (2 ** 32)
    except ImportError:
        return fallback

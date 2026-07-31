"""
training/train.py

CLI entry point. Implements exactly the command surface already committed
to in research/reproducibility.md Section 8:

    python training/train.py --config configs/config.yaml --model agrivisionnet --seed 42 --run-name ours_s42
    python training/train.py --config configs/config.yaml --model agrivisionnet --ablation shared_fusion --seed 42 --run-name ablation_A1_s42

Requires torch/timm -- UNVERIFIED in this authoring sandbox, same caveat as
training/trainer.py. The CLI argument parsing and config-merging logic
below has no torch dependency and IS actually exercised in this sandbox
(see tests/test_training/test_train_cli.py), but the full run is not.

--model selects which architecture actually trains: the three single-
backbone baselines (research/experiments.md Section 1) share this same
entry point rather than a separate script, so every run in the comparison
table goes through identical data loading / seeding / logging code -- the
only difference is which model gets built.

--ablation is currently a documented, PARTIALLY UNIMPLEMENTED interface:
A2/A3/A4 map onto existing config fields and work today. A1 (shared
fusion) and A5 (without evidential learning) require new model code not
yet written (see research/experiments.md's ablation table) -- passing
those two currently raises NotImplementedError with a message pointing at
that gap, rather than silently running the wrong model.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


ABLATION_OVERRIDES = {
    # A2: task-aware fusion in isolation, discounting off
    "task_aware_only": {"loss": {"discount_severity_by_disease_uncertainty": False}},
    # A3: hierarchical discounting in isolation -- requires shared-gate mode,
    # which doesn't exist yet either (see NOT_YET_IMPLEMENTED below); listed
    # here for documentation even though it currently also raises.
    "hierarchical_only": {"loss": {"discount_severity_by_disease_uncertainty": True}},
    # A4: full model as already built -- no override needed, but named
    # explicitly so `--ablation full` is a valid, self-documenting no-op
    # rather than requiring the person to omit --ablation to mean "full".
    "full": {},
}

NOT_YET_IMPLEMENTED_ABLATIONS = {
    "shared_fusion": (
        "A1 (shared fusion) requires a SharedReliabilityGate class as an "
        "alternative to models/fusion/task_hierarchy_fusion.py's "
        "PerTaskReliabilityGate -- not yet built (see "
        "research/experiments.md's A1 row). This is an architecture change "
        "requiring approval per the Phase 1 freeze, not a Phase 3 task."
    ),
    "no_evidential": (
        "A5 (without evidential learning) requires a parallel non-evidential "
        "head/gate path -- not yet built (see research/experiments.md's A5 "
        "row). Same freeze caveat as A1."
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AgriVisionNet training entry point")
    p.add_argument("--config", type=str, required=True, help="path to configs/config.yaml")
    p.add_argument(
        "--model", type=str, required=True,
        choices=["resnet50", "efficientnet_b0", "swin_tiny", "agrivisionnet"],
        help="which architecture to train -- see research/experiments.md Section 1",
    )
    p.add_argument("--seed", type=int, default=None, help="overrides config's project.seed if given")
    p.add_argument("--run-name", type=str, required=True, help="outputs/runs/<run-name>/")
    p.add_argument(
        "--ablation", type=str, default="full",
        choices=list(ABLATION_OVERRIDES.keys()) + list(NOT_YET_IMPLEMENTED_ABLATIONS.keys()),
        help="only meaningful when --model agrivisionnet -- see research/experiments.md",
    )
    p.add_argument("--resume", action="store_true", help="resume from outputs/runs/<run-name>/checkpoint_latest.pt")
    return p


def resolve_config(args: argparse.Namespace) -> dict:
    """Loads configs/config.yaml and applies CLI overrides, producing the
    exact dict that gets written to resolved_config.yaml
    (training/checkpoint_manager.py) -- this function is the single place
    CLI args turn into config values, so that traceability guarantee holds.
    No torch dependency -- actually exercised in this sandbox."""
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"--config path does not exist: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)  # never mutate a shared/cached load

    if args.seed is not None:
        cfg["project"]["seed"] = args.seed

    if args.ablation in NOT_YET_IMPLEMENTED_ABLATIONS:
        raise NotImplementedError(
            f"--ablation {args.ablation}: {NOT_YET_IMPLEMENTED_ABLATIONS[args.ablation]}"
        )

    overrides = ABLATION_OVERRIDES.get(args.ablation, {})
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)

    cfg["_cli"] = {  # kept in resolved_config.yaml for full traceability, not consumed by the model
        "model": args.model,
        "run_name": args.run_name,
        "ablation": args.ablation,
        "resume": args.resume,
    }

    return cfg


def main():
    args = build_arg_parser().parse_args()
    cfg = resolve_config(args)

    # Deferred imports: everything below needs torch/timm, which this
    # authoring sandbox doesn't have. Keeping CLI parsing and config
    # resolution above import-safe means `--help` and config validation
    # work even without torch installed, and means
    # tests/test_training/test_train_cli.py (torch-free) can exercise
    # resolve_config() directly without needing torch either.
    from training.trainer import Trainer
    from datasets.unified import UnifiedCropDataset
    from datasets.splitting import stratified_split
    from datasets.transforms import build_train_transform, build_eval_transform
    from torch.utils.data import DataLoader, Subset

    full_dataset = UnifiedCropDataset(
        cfg["data"]["datasets"],
        transform=None,  # transform applied per-split below, not shared
        severity_annotations_csv=cfg["data"].get("severity_annotations_csv"),
    )
    split = stratified_split(full_dataset.all_samples(), seed=cfg["project"]["seed"])

    train_ds = UnifiedCropDataset(
        cfg["data"]["datasets"], transform=build_train_transform(cfg),
        label_spaces=full_dataset.label_spaces(),
        severity_annotations_csv=cfg["data"].get("severity_annotations_csv"),
    )
    eval_ds = UnifiedCropDataset(
        cfg["data"]["datasets"], transform=build_eval_transform(cfg),
        label_spaces=full_dataset.label_spaces(),
        severity_annotations_csv=cfg["data"].get("severity_annotations_csv"),
    )

    train_loader = DataLoader(
        Subset(train_ds, split.train_indices),
        batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, split.val_indices),
        batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    if args.model != "agrivisionnet":
        raise NotImplementedError(
            f"--model {args.model}: the 3 single-backbone baselines "
            f"(research/experiments.md Section 1) need a separate thin "
            f"model-building path (plain softmax head on disease label only, "
            f"per that doc's stated design decision) -- not yet built. Only "
            f"--model agrivisionnet is currently runnable."
        )

    trainer = Trainer(cfg, run_name=args.run_name, resume=args.resume)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

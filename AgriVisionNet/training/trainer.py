"""
training/trainer.py

The actual training loop -- requires torch/timm, NEITHER installed in this
authoring sandbox (see README.md). Written completely (no placeholders,
per the project's standing instruction) but UNVERIFIED here: run
`pytest tests/test_training/test_trainer.py` locally, with real torch, GPU
optional but recommended, before trusting this against a real dataset.

Wraps together everything already built and VERIFIED in isolation:
- models/agrivisionnet.py (Phase 1, unverified end-to-end but math-checked)
- losses/evidential_loss.py (Phase 1)
- datasets/unified.py (Phase 2, verified)
- training/seeding.py, early_stopping.py, lr_schedule.py, checkpoint_manager.py (Phase 3, all verified above)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.agrivisionnet import build_model
from losses.evidential_loss import build_loss
from training.checkpoint_manager import CheckpointManager
from training.early_stopping import EarlyStopping
from training.lr_schedule import cosine_warmup_multiplier
from training.seeding import set_seed, seed_worker


class Trainer:
    def __init__(self, cfg: dict, run_name: str, resume: bool = False):
        """
        Args:
            cfg: fully-resolved config dict (already merged with any CLI
                overrides -- see training/train.py's CLI, which is
                responsible for producing this dict).
            run_name: identifies outputs/runs/<run_name>/ -- matches the
                CLI interface already committed to in
                research/reproducibility.md Section 8.
            resume: if True, load the latest checkpoint from this run's
                directory and continue, rather than starting fresh.
        """
        self.cfg = cfg
        self.run_name = run_name
        self.run_dir = Path(cfg["project"]["output_dir"]) / "runs" / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        set_seed(cfg["project"]["seed"])

        self.device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
        if cfg["project"]["device"] == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn("configs/config.yaml requests device='cuda' but no GPU is available -- falling back to CPU. Training will be extremely slow; this is expected in this authoring sandbox specifically.")

        self.model = build_model(cfg).to(self.device)
        self.loss_fn = build_loss(cfg).to(self.device)

        self.optimizer = self._build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["mixed_precision"])

        self.early_stopping = EarlyStopping(
            patience=cfg["training"]["early_stopping_patience"], mode="min"
        )
        self.checkpoint_manager = CheckpointManager(
            run_dir=self.run_dir, monitor_mode="min", keep_last_n=3
        )

        self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.start_epoch = 0

        if resume:
            self._load_latest_checkpoint()

        # Traceability: write the exact resolved config (post any CLI
        # overrides) used for this run, per research/reproducibility.md
        # Section 8. Done AFTER resume-loading so a resumed run's config
        # file reflects what's actually running now, not what was
        # originally launched, if it changed.
        self.checkpoint_manager.save_resolved_config(cfg)

    def _build_optimizer(self):
        opt_name = self.cfg["training"]["optimizer"].lower()
        lr = self.cfg["training"]["lr"]
        wd = self.cfg["training"]["weight_decay"]
        if opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        raise NotImplementedError(
            f"training.optimizer='{opt_name}' not implemented -- only 'adamw' "
            f"is currently supported. Add a branch here rather than silently "
            f"falling back to a default optimizer the config didn't request."
        )

    def _lr_lambda(self, epoch: int) -> float:
        return cosine_warmup_multiplier(
            epoch=epoch,
            total_epochs=self.cfg["training"]["epochs"],
            warmup_epochs=self.cfg["training"]["warmup_epochs"],
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        # fast-forward the scheduler if resuming mid-run
        for _ in range(self.start_epoch):
            scheduler.step()

        for epoch in range(self.start_epoch, self.cfg["training"]["epochs"]):
            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, epoch, train=True)
            val_metrics = self._run_epoch(val_loader, epoch, train=False)
            scheduler.step()
            epoch_time = time.time() - t0

            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)

            plan = self.checkpoint_manager.decide(epoch, val_metrics["total_loss"])
            self._save_checkpoints(epoch, plan)

            should_stop = self.early_stopping.step(val_metrics["total_loss"], epoch)
            if should_stop:
                print(
                    f"[Trainer] Early stopping triggered at epoch {epoch} "
                    f"(best epoch: {self.early_stopping.best_epoch}, "
                    f"best val_loss: {self.early_stopping.best_value:.4f})"
                )
                break

        self.tb_writer.close()

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> dict:
        self.model.train(train)
        running = {}
        n_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = {
                "crop_label": batch["crop_label"].to(self.device),
                "disease_label": batch["disease_label"].to(self.device),
                "severity_label": batch["severity_label"].to(self.device),
            }

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.cfg["training"]["mixed_precision"]):
                    outputs = self.model(
                        images,
                        discount_severity=self.cfg["loss"]["discount_severity_by_disease_uncertainty"],
                    )
                    # epoch is required for evidential_classification_loss's KL
                    # annealing coefficient (models/evidential.py) -- omitting
                    # it was caught here before it became a runtime TypeError
                    # the first time this was actually run with torch installed.
                    losses = self.loss_fn(outputs, targets, epoch)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(losses["total_loss"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["training"]["grad_clip_norm"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().cpu())
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def _log_epoch(self, epoch, train_metrics, val_metrics, epoch_time):
        print(
            f"[epoch {epoch}] train_loss={train_metrics.get('total_loss', float('nan')):.4f} "
            f"val_loss={val_metrics.get('total_loss', float('nan')):.4f} "
            f"time={epoch_time:.1f}s"
        )
        for k, v in train_metrics.items():
            self.tb_writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.tb_writer.add_scalar(f"val/{k}", v, epoch)

    def _save_checkpoints(self, epoch: int, plan: dict):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "early_stopping_state": self.early_stopping.state_dict(),
            "checkpoint_manager_state": self.checkpoint_manager.state_dict(),
            "config": self.cfg,
        }
        if plan["save_latest"]:
            torch.save(state, self.run_dir / self.checkpoint_manager.latest_checkpoint_filename())
        if plan["save_best"]:
            torch.save(state, self.run_dir / self.checkpoint_manager.best_checkpoint_filename())

        epoch_path = self.run_dir / self.checkpoint_manager.checkpoint_filename(epoch)
        torch.save(state, epoch_path)
        for old_epoch in plan["epochs_to_delete"]:
            old_path = self.run_dir / self.checkpoint_manager.checkpoint_filename(old_epoch)
            if old_path.exists():
                old_path.unlink()

    def _load_latest_checkpoint(self):
        latest_path = self.run_dir / self.checkpoint_manager.latest_checkpoint_filename()
        if not latest_path.exists():
            print(f"[Trainer] resume=True but no checkpoint found at {latest_path} -- starting fresh.")
            return

        state = torch.load(latest_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.early_stopping = EarlyStopping.from_state_dict(state["early_stopping_state"])
        self.checkpoint_manager = CheckpointManager.from_state_dict(
            self.run_dir, state["checkpoint_manager_state"]
        )
        self.start_epoch = state["epoch"] + 1
        print(f"[Trainer] resumed from epoch {state['epoch']}, continuing at epoch {self.start_epoch}")

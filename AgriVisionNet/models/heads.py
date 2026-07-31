"""
models/heads.py

Multi-task EVIDENTIAL heads. Each task (crop, disease, severity) receives
its OWN fused vector from TaskHierarchyEvidentialFusion (not a shared
trunk), and outputs a Dirichlet distribution (models/evidential.py)
instead of softmax logits.

WHY EVIDENTIAL HEADS SPECIFICALLY HERE
-----------------------------------------
This is required by the frozen architecture, not a stylistic choice: Stage 5
(hierarchical discounting, applied here since it needs the disease head's
output) only makes sense if severity's output is itself a Dirichlet that CAN
be discounted. A softmax severity head would have nothing for `ds_discount`
to operate on.

HIERARCHICAL DISCOUNTING (Stage 5, Shafer 1976)
---------------------------------------------------
severity's belief/uncertainty is discounted using the DISEASE head's own
uncertainty as the discount rate:

    r = disease_dirichlet.uncertainty                (B,)
    severity_belief', severity_uncertainty' = ds_discount(
        severity_dirichlet.belief, severity_dirichlet.uncertainty, r
    )

This is applied to the belief/uncertainty used for DOWNSTREAM decisions
(confidence display, unknown-disease gating, recommendation confidence) --
the RAW (undiscounted) severity Dirichlet is what the evidential loss trains
against, so gradient flow through Stage 4's severity gate is not disturbed
by a non-differentiable dependency ordering concern. Both raw and
discounted severity outputs are returned; training uses raw, inference/
downstream consumers use discounted (see ablation note in
tests/test_fusion.py: with vs. without discounting compares these two).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.evidential import EvidentialHead, ds_discount


class MultiTaskEvidentialHeads(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        shared_dim = cfg["heads"]["shared_dim"]
        crop_cfg = cfg["heads"]["crop"]
        disease_cfg = cfg["heads"]["disease"]
        severity_cfg = cfg["heads"]["severity"]

        self.crop_head = EvidentialHead(
            shared_dim, crop_cfg["num_classes"], crop_cfg["hidden_dim"], crop_cfg["dropout"]
        )
        self.disease_head = EvidentialHead(
            shared_dim, disease_cfg["num_classes"], disease_cfg["hidden_dim"], disease_cfg["dropout"]
        )
        self.severity_head = EvidentialHead(
            shared_dim, severity_cfg["num_classes"], severity_cfg["hidden_dim"], severity_cfg["dropout"]
        )

    def forward(self, fused: dict, discount_severity: bool = True) -> dict:
        """
        Args:
            fused: dict with 'crop', 'disease', 'severity' -> (B, shared_dim),
                   as produced by TaskHierarchyEvidentialFusion
            discount_severity: whether to apply Stage 5 DS discounting to the
                   severity output used for downstream decisions (ablation switch)
        """
        crop_dirichlet = self.crop_head(fused["crop"])
        disease_dirichlet = self.disease_head(fused["disease"])
        severity_dirichlet = self.severity_head(fused["severity"])

        result = {
            "crop_dirichlet": crop_dirichlet,
            "disease_dirichlet": disease_dirichlet,
            "severity_dirichlet": severity_dirichlet,          # RAW -- used for training loss
        }

        if discount_severity:
            disc_belief, disc_uncertainty = ds_discount(
                severity_dirichlet.belief,
                severity_dirichlet.uncertainty,
                discount_rate=disease_dirichlet.uncertainty,
            )
            result["severity_belief_discounted"] = disc_belief
            result["severity_uncertainty_discounted"] = disc_uncertainty
        else:
            result["severity_belief_discounted"] = severity_dirichlet.belief
            result["severity_uncertainty_discounted"] = severity_dirichlet.uncertainty

        return result

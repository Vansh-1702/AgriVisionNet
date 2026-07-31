"""
models/agrivisionnet.py

Top-level model: DualBranchBackbone -> TaskHierarchyEvidentialFusion -> MultiTaskEvidentialHeads.

Kept thin by design (see Phase 1 rationale) so each stage remains
independently unit-testable and ablatable:
    - backbone.py                         (standard EfficientNet-B0 + Swin-Tiny, not novel)
    - fusion/task_hierarchy_fusion.py      (frozen architecture's specific contribution)
    - heads.py                            (evidential heads + hierarchical discounting)

Extension point for future modalities (weather/IoT/soil/drone/LLM, per the
original brief): add a new encoder branch, extend
TaskHierarchyEvidentialFusion's Stage 0-4 to accept an additional pooled
vector, without touching backbone.py or heads.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone import DualBranchBackbone
from models.fusion.task_hierarchy_fusion import TaskHierarchyEvidentialFusion
from models.heads import MultiTaskEvidentialHeads


class AgriVisionNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = DualBranchBackbone(cfg)

        fusion_cfg = cfg["fusion"]
        disease_num_classes = cfg["heads"]["disease"]["num_classes"]

        self.fusion = TaskHierarchyEvidentialFusion(
            cnn_channels=self.backbone.cnn_channels,
            swin_channels=self.backbone.swin_channels,
            projected_dim=fusion_cfg["projected_dim"],
            se_reduction=fusion_cfg["se_reduction"],
            spatial_kernel=fusion_cfg["spatial_kernel"],
            gate_hidden_dim=fusion_cfg["gate_hidden_dim"],
            dropout=fusion_cfg["dropout"],
            use_residual=fusion_cfg["use_residual"],
            auxiliary_evidence_dim=fusion_cfg["auxiliary_evidence_dim"],
            num_disease_classes=disease_num_classes,
        )

        assert cfg["heads"]["shared_dim"] == fusion_cfg["projected_dim"], (
            "heads.shared_dim must equal fusion.projected_dim in config.yaml"
        )
        self.heads = MultiTaskEvidentialHeads(cfg)

    def forward(self, x: torch.Tensor, discount_severity: bool = True, return_aux: bool = False):
        cnn_feat, swin_feat = self.backbone(x)
        fused, fusion_aux = self.fusion(cnn_feat, swin_feat)
        outputs = self.heads(fused, discount_severity=discount_severity)

        # auxiliary branch-reliability Dirichlets are exposed so they can ALSO
        # be supervised with the disease label (a small extra loss term) --
        # otherwise nothing teaches them to be well-calibrated in the first
        # place, and the Stage-4 gate would be conditioning on noise. This is
        # required for the "auxiliary evidence must be supervised" experiment
        # in tests/test_fusion.py.
        outputs["aux_dirichlet_cnn"] = fusion_aux["aux_dirichlet_cnn"]
        outputs["aux_dirichlet_swin"] = fusion_aux["aux_dirichlet_swin"]

        if return_aux:
            outputs["fusion_aux"] = fusion_aux
            outputs["raw_cnn_feat"] = cnn_feat      # kept for Grad-CAM hooks
            outputs["raw_swin_feat"] = swin_feat    # kept for attention-rollout hooks

        return outputs


def build_model(cfg: dict) -> AgriVisionNet:
    """Factory used by training/evaluation/inference scripts."""
    return AgriVisionNet(cfg)


if __name__ == "__main__":
    # Quick shape sanity check -- run with: python -m models.agrivisionnet
    # NOTE: requires torch/timm installed; not executed in the authoring
    # sandbox (no GPU/network access there). Run this locally before
    # building Phase 2 on top of it.
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    dummy = torch.randn(2, 3, cfg["backbone"]["input_size"], cfg["backbone"]["input_size"])
    out = model(dummy, return_aux=True)

    print("crop uncertainty:", out["crop_dirichlet"].uncertainty.shape)
    print("disease uncertainty:", out["disease_dirichlet"].uncertainty.shape)
    print("severity uncertainty (discounted):", out["severity_uncertainty_discounted"].shape)
    print("per-task gate (disease):", out["fusion_aux"]["gate_values"]["disease"].shape)

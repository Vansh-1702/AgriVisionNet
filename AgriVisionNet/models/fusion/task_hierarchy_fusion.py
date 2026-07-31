"""
models/fusion/task_hierarchy_fusion.py

Task- and Hierarchy-Aware Evidential Fusion — the frozen architecture.

WHY THIS EXISTS
-----------------
Two branches (CNN: local texture bias, Swin: global context bias) need to be
combined into a single representation per task. A single shared fusion gate
(as in MMTM/AFF) assumes every downstream task trusts each branch equally —
but crop identification (whole-leaf shape, venation, overall color) is a
different visual problem from disease localization (a small textured lesion
patch), so there is no reason to expect one gate serves both tasks equally
well. Separately, severity is only meaningful GIVEN a disease identity: a
"severe" prediction is not interpretable unless the disease head is also
reasonably confident about what disease it is looking at.

PRIOR WORK AND HOW WE DIFFER
-------------------------------
- Joze et al. "MMTM." CVPR 2020 — per-sample, per-channel gate from pooled
  multimodal stats. We differ by (a) conditioning the gate on EVIDENTIAL
  RELIABILITY (1 - Dirichlet uncertainty), not raw pooled feature
  statistics, and (b) using a SEPARATE gate per task, not one shared gate.
- Dai et al. "AFF/iAFF." WACV 2021 — channel-attention-weighted sum of two
  feature maps. Same feature-level fusion family; same two deltas as above.
- Han et al. "TMC." ICLR 2021 — evidential combination via Dempster-Shafer,
  but at the DECISION level, across multiple views, single-task. We fuse at
  the FEATURE level, across two branches sharing one input image, across
  THREE tasks with an explicit inter-task hierarchy — none of which TMC/ETMC
  address.
- Shafer, 1976 — the Dempster-Shafer DISCOUNTING operator (not TMC's
  combination rule) is reused here to implement "distrust severity's
  evidence in proportion to disease's uncertainty."

WHAT IS AND ISN'T CLAIMED
----------------------------
NOT claimed as novel: channel/spatial attention (Stages 1-2), Dirichlet
evidential heads themselves, the Dempster-Shafer discounting formula.
CLAIMED as the project's specific, modest, checkable delta: applying
evidential-reliability-conditioned, per-TASK gating (Stage 4) plus
hierarchical discounting (Stage 5) together, at the feature level, in a
multi-task hybrid CNN-Transformer setting. See configs/config.yaml header
and research/design_rationale.md for the full positioning.

STAGES
--------
Stage 0  Project both branches to a common dim d (1x1 conv), align spatial size.
Stage 1  Channel attention (SE), per branch — standard, not novel.
Stage 2  Spatial attention (CBAM-style), per branch — standard, not novel.
Stage 3  Auxiliary evidential heads on each branch's pooled features, over
         the DISEASE label space only (the hardest, most informative task),
         producing per-branch uncertainty u_cnn, u_swin -> reliability
         r_cnn = 1 - u_cnn, r_swin = 1 - u_swin.
Stage 4  Per-TASK reliability gate: for each of {crop, disease, severity},
         a small MLP maps [r_cnn, r_swin] -> a per-channel gate g_task in
         (0,1)^d, producing a task-specific fused vector:
             f_task = g_task * pooled_cnn + (1 - g_task) * pooled_swin
                      + Proj_task([pooled_cnn ; pooled_swin])   (residual)
Stage 5  (Applied downstream, in AgriVisionNet.forward, AFTER the task
         evidential heads run on f_task): severity's belief/uncertainty is
         Dempster-Shafer-discounted using the DISEASE head's own final
         uncertainty as the discount rate. Kept out of this module because
         it needs the disease head's OUTPUT, which doesn't exist until
         after Stage 4's fused vectors are passed through the heads.

ABLATIONS THIS DESIGN REQUIRES (tests/test_fusion.py + training/evaluate.py)
--------------------------------------------------------------------------------
  - Shared gate (one MLP for all 3 tasks) vs. per-task gate (this module) —
    isolates the "task-aware" claim.
  - With vs. without Stage 5 hierarchical discounting — isolates the
    "hierarchy-aware" claim.
  - Gate conditioned on reliability (this module) vs. gate conditioned on
    raw pooled features (MMTM-style) — isolates "evidential-conditioning"
    specifically, not just "having a gate at all."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion.attention_blocks import ChannelAttention, SpatialAttention
from models.evidential import EvidentialHead


TASK_NAMES = ("crop", "disease", "severity")


class PerTaskReliabilityGate(nn.Module):
    """One small MLP per task, mapping [r_cnn, r_swin] (2 scalars) to a
    per-channel gate vector g in (0,1)^d. Kept intentionally tiny (2 -> d)
    since the input is just two reliability scalars, not full features —
    the whole point is that trust, not raw magnitude, drives the mix."""

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim),
            )
            for task in TASK_NAMES
        })

    def forward(self, reliability_cnn: torch.Tensor, reliability_swin: torch.Tensor) -> dict:
        r = torch.stack([reliability_cnn, reliability_swin], dim=-1)  # (B, 2)
        return {task: torch.sigmoid(self.gates[task](r)) for task in TASK_NAMES}


class TaskHierarchyEvidentialFusion(nn.Module):
    def __init__(
        self,
        cnn_channels: int,
        swin_channels: int,
        projected_dim: int = 512,
        se_reduction: int = 16,
        spatial_kernel: int = 7,
        gate_hidden_dim: int = 128,
        dropout: float = 0.1,
        use_residual: bool = True,
        auxiliary_evidence_dim: int = 128,
        num_disease_classes: int = 38,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.projected_dim = projected_dim

        # Stage 0
        self.proj_cnn = nn.Conv2d(cnn_channels, projected_dim, kernel_size=1)
        self.proj_swin = nn.Conv2d(swin_channels, projected_dim, kernel_size=1)

        # Stage 1
        self.channel_attn_cnn = ChannelAttention(projected_dim, se_reduction)
        self.channel_attn_swin = ChannelAttention(projected_dim, se_reduction)

        # Stage 2
        self.spatial_attn_cnn = SpatialAttention(spatial_kernel)
        self.spatial_attn_swin = SpatialAttention(spatial_kernel)

        # Stage 3 — auxiliary reliability heads (DISEASE label space only)
        self.aux_evidence_cnn = EvidentialHead(
            projected_dim, num_disease_classes, hidden_dim=auxiliary_evidence_dim
        )
        self.aux_evidence_swin = EvidentialHead(
            projected_dim, num_disease_classes, hidden_dim=auxiliary_evidence_dim
        )

        # Stage 4 — per-task reliability gate + residual projection
        self.gate = PerTaskReliabilityGate(projected_dim, gate_hidden_dim)
        self.residual_proj = nn.ModuleDict({
            task: nn.Linear(projected_dim * 2, projected_dim) for task in TASK_NAMES
        })

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.ModuleDict({task: nn.LayerNorm(projected_dim) for task in TASK_NAMES})

    def forward(self, cnn_feat: torch.Tensor, swin_feat: torch.Tensor):
        """
        Args:
            cnn_feat:  (B, C_cnn, h1, w1)
            swin_feat: (B, C_swin, h2, w2)
        Returns:
            fused: dict[task] -> (B, projected_dim) task-specific fused vector
            aux: dict of intermediate values needed for logging/explainability/OOD
        """
        # Stage 0
        f_c = self.proj_cnn(cnn_feat)
        f_s = self.proj_swin(swin_feat)
        if f_c.shape[-2:] != f_s.shape[-2:]:
            f_s = F.interpolate(f_s, size=f_c.shape[-2:], mode="bilinear", align_corners=False)

        # Stage 1
        f_c = self.channel_attn_cnn(f_c)
        f_s = self.channel_attn_swin(f_s)

        # Stage 2
        f_c, mask_c = self.spatial_attn_cnn(f_c)
        f_s, mask_s = self.spatial_attn_swin(f_s)

        pooled_c = self.pool(f_c).flatten(1)  # (B, d)
        pooled_s = self.pool(f_s).flatten(1)  # (B, d)

        # Stage 3 — branch reliability via auxiliary evidential heads
        aux_dirichlet_cnn = self.aux_evidence_cnn(pooled_c)
        aux_dirichlet_swin = self.aux_evidence_swin(pooled_s)
        reliability_cnn = 1.0 - aux_dirichlet_cnn.uncertainty   # (B,)
        reliability_swin = 1.0 - aux_dirichlet_swin.uncertainty  # (B,)

        # Stage 4 — per-task gating + residual
        gates = self.gate(reliability_cnn, reliability_swin)
        fused = {}
        for task in TASK_NAMES:
            g = gates[task]
            gated = g * pooled_c + (1.0 - g) * pooled_s
            residual = self.residual_proj[task](torch.cat([pooled_c, pooled_s], dim=-1))
            f_task = gated + residual if self.use_residual else gated
            fused[task] = self.dropout(self.out_norm[task](f_task))

        aux = {
            "spatial_mask_cnn": mask_c,             # (B,1,h,w) for explainability
            "spatial_mask_swin": mask_s,             # (B,1,h,w) for explainability
            "gate_values": {t: gates[t].detach() for t in TASK_NAMES},  # per-task gate, for ablation logging
            "reliability_cnn": reliability_cnn.detach(),
            "reliability_swin": reliability_swin.detach(),
            "aux_dirichlet_cnn": aux_dirichlet_cnn,   # exposed so the auxiliary heads can ALSO be
            "aux_dirichlet_swin": aux_dirichlet_swin, # supervised with a disease label (see loss)
        }
        return fused, aux

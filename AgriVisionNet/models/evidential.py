"""
models/evidential.py

Evidential Deep Learning (EDL) toolbox.

WHY THIS EXISTS
-----------------
A standard softmax head outputs a point estimate over classes with no
distinction between "confident and correct," "confident and wrong," and "not
enough evidence to say." EDL (Sensoy, Kaplan & Kandemir, NeurIPS 2018)
replaces softmax with a **second-order** distribution: instead of predicting
class probabilities directly, the network predicts the parameters of a
Dirichlet distribution OVER the categorical class probabilities. This gives
a single-forward-pass, closed-form measure of predictive uncertainty (no
sampling, unlike MC Dropout or Deep Ensembles), which is exactly the signal
our fusion module needs to decide how much to trust each branch, and exactly
the signal Phase 5 (unknown-disease detection) needs to reject out-of-scope
inputs.

PRIOR WORK — do not claim this mechanism as novel
----------------------------------------------------
- Sensoy, Kaplan, Kandemir. "Evidential Deep Learning to Quantify
  Classification Uncertainty." NeurIPS 2018. (the Dirichlet-evidence head
  and type-II MLE loss implemented below)
- Malinin & Gales. "Predictive Uncertainty Estimation via Prior Networks."
  NeurIPS 2018. (alternative Dirichlet-Prior formulation, separates
  aleatoric/epistemic uncertainty more explicitly — noted as a documented
  alternative, not implemented here to keep one consistent formulation)
- Han et al. "Trusted Multi-View Classification." ICLR 2021 (TMC). Combines
  MULTIPLE per-view Dirichlets via the Dempster-Shafer rule at the DECISION
  level. Our use of Dempster-Shafer DISCOUNTING (not their combination rule)
  for hierarchical task discounting is a different operator, applied to a
  single Dirichlet output conditioned on another task's uncertainty, not to
  combine several views' beliefs into one.
- Shafer. "A Mathematical Theory of Evidence." 1976. (the discounting
  operator itself, `ds_discount` below)

HOW OUR USE DIFFERS (see configs/config.yaml header for full positioning)
--------------------------------------------------------------------------
This module is used in two places in AgriVisionNet, both DIFFERENT from how
TMC uses it:
  1. `EvidentialHead` as a lightweight AUXILIARY head on each branch's pooled
     features (CNN, Swin) purely to estimate branch RELIABILITY — feeding a
     per-task fusion GATE (models/fusion/task_hierarchy_fusion.py) rather
     than being combined via Dempster-Shafer at the decision level.
  2. `EvidentialHead` as the FINAL task heads (crop/disease/severity) on the
     fused, task-specific representation, with severity's output further
     modified by `ds_discount` conditioned on the disease head's uncertainty
     — a hierarchical use of discounting that TMC/ETMC do not do (they are
     single-task).

VALIDATION EXPERIMENTS THIS MODULE MUST SUPPORT (see tests/test_evidential.py)
---------------------------------------------------------------------------------
  - Evidence non-negativity: alpha_k >= 1 for all k, always (softplus + 1).
  - Uncertainty mass u in (0, 1], and u -> 1 as evidence -> 0 (untrained /
    OOD network should default to "I don't know," not a confident guess).
  - `ds_discount` recovers the identity when discount_rate=0, and drives
    uncertainty toward 1 as discount_rate -> 1 (full discounting = total
    ignorance), matching Shafer's original operator algebraically.
  - KL-annealing coefficient is monotonic non-decreasing across epochs and
    saturates at 1.0, never exceeds it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DirichletOutput:
    """Container for one evidential head's output on a batch.

    Attributes:
        evidence: (B, K) non-negative evidence per class, e_k = softplus(logit_k)
        alpha:    (B, K) Dirichlet concentration params, alpha_k = e_k + 1
        strength: (B,)   total Dirichlet strength S = sum_k alpha_k
        belief:   (B, K) belief mass per class, b_k = e_k / S  (sums to 1-u)
        prob:     (B, K) expected categorical probability, p_k = alpha_k / S
        uncertainty: (B,) total uncertainty mass, u = K / S, in (0, 1]
    """
    evidence: torch.Tensor
    alpha: torch.Tensor
    strength: torch.Tensor
    belief: torch.Tensor
    prob: torch.Tensor
    uncertainty: torch.Tensor


class EvidentialHead(nn.Module):
    """Maps a feature vector to Dirichlet evidence over `num_classes`.

    Architecture is deliberately a plain linear layer + softplus: the
    contribution here is not the head's capacity but where/how its output
    (uncertainty) is consumed elsewhere in the network. Keeping the head
    itself minimal also keeps the auxiliary reliability heads cheap (they
    run twice per forward pass, once per branch).
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> DirichletOutput:
        logits = self.net(x)
        # softplus keeps evidence >= 0 everywhere, with a smoother gradient
        # near 0 than ReLU (matters early in training when most evidence is
        # near zero) -- this is the activation used in Sensoy et al. 2018.
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)                      # (B,)
        belief = evidence / strength.unsqueeze(-1)          # (B, K)
        prob = alpha / strength.unsqueeze(-1)                # (B, K), expected prob under Dirichlet
        uncertainty = self.num_classes / strength            # (B,)

        return DirichletOutput(
            evidence=evidence, alpha=alpha, strength=strength,
            belief=belief, prob=prob, uncertainty=uncertainty,
        )


def ds_discount(belief: torch.Tensor, uncertainty: torch.Tensor,
                 discount_rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dempster-Shafer discounting operator (Shafer, 1976).

    Attenuates a belief function by a per-sample discount rate r in [0,1],
    representing "how much should we distrust this source of evidence."
    Used here with r = disease-head uncertainty, to discount the SEVERITY
    head's belief: if the model isn't sure what disease this is, it should
    not be confident about the disease's severity either.

        b'_k = (1 - r) * b_k                      for each class k
        u'   = r + (1 - r) * u

    Properties (validated in tests/test_evidential.py):
        r=0  -> (b', u') == (b, u)          no discounting, identity
        r=1  -> (b', u') == (0, 1)          total ignorance regardless of b, u
        b' always sums to (1-u') exactly, preserving the Dirichlet mass
        budget (belief + uncertainty == 1), i.e. it stays a valid belief
        function after discounting.

    Args:
        belief: (B, K) belief masses from a DirichletOutput
        uncertainty: (B,) uncertainty mass from the SAME DirichletOutput
        discount_rate: (B,) in [0, 1], typically another task's uncertainty
    Returns:
        (discounted_belief (B,K), discounted_uncertainty (B,))
    """
    r = discount_rate.unsqueeze(-1)                 # (B, 1)
    discounted_belief = (1.0 - r) * belief
    discounted_uncertainty = discount_rate + (1.0 - discount_rate) * uncertainty
    return discounted_belief, discounted_uncertainty


def kl_annealing_coefficient(epoch: int, annealing_steps: int) -> float:
    """Linear ramp 0 -> 1 over `annealing_steps` epochs, then clamped at 1.

    Sensoy et al. anneal the KL regularizer so the network is not punished
    for holding non-zero evidence before it has learned anything useful —
    without annealing, the KL term dominates early training and evidence
    collapses to zero everywhere (the network never learns to be confident
    even when it should be).
    """
    if annealing_steps <= 0:
        return 1.0
    return float(min(1.0, epoch / annealing_steps))

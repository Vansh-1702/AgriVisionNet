"""
tests/test_evidential.py

Unit tests for models/evidential.py.

NOTE ON EXECUTION: this authoring sandbox has no torch installed (no
network access to pip install it). These tests are written to run under
`pytest` in a real environment (`pip install -r requirements.txt && pytest
tests/`) and are skipped here with a clear reason rather than silently
passing, so CI failure vs. "torch missing" is never ambiguous. The
Dempster-Shafer algebraic properties these tests also check were
independently verified with plain-Python arithmetic (no torch dependency)
during authoring -- see the sandbox transcript / research/design_rationale.md.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed in this environment")

from models.evidential import EvidentialHead, ds_discount, kl_annealing_coefficient  # noqa: E402


class TestEvidentialHead:
    def test_evidence_non_negative(self):
        """Evidence must be >= 0 for arbitrary input, including large negative logits."""
        head = EvidentialHead(in_dim=16, num_classes=5)
        head.eval()
        x = torch.randn(8, 16) * 100  # push toward extreme logits
        out = head(x)
        assert (out.evidence >= 0).all()
        assert (out.alpha >= 1.0).all(), "alpha = evidence + 1 must always be >= 1"

    def test_uncertainty_range(self):
        """Uncertainty mass must lie in (0, 1] for any finite input."""
        head = EvidentialHead(in_dim=16, num_classes=10)
        head.eval()
        x = torch.randn(32, 16)
        out = head(x)
        assert (out.uncertainty > 0).all()
        assert (out.uncertainty <= 1.0).all()

    def test_zero_evidence_gives_max_uncertainty(self):
        """An untrained/zero-evidence head should default to maximal
        uncertainty (u = K/S = K/K = 1), not a confident guess -- this is
        the core safety property EDL is chosen for."""
        head = EvidentialHead(in_dim=4, num_classes=6)
        with torch.no_grad():
            head.net.weight.zero_()
            head.net.bias.fill_(-1000.0)  # force softplus(logit) ~= 0 evidence
        x = torch.randn(4, 4)
        out = head(x)
        assert torch.allclose(out.uncertainty, torch.ones_like(out.uncertainty), atol=1e-4)

    def test_belief_plus_uncertainty_is_valid_mass_function(self):
        """sum_k(belief_k) + uncertainty == 1 must hold exactly (Dirichlet mass budget)."""
        head = EvidentialHead(in_dim=16, num_classes=7)
        head.eval()
        x = torch.randn(8, 16)
        out = head(x)
        total = out.belief.sum(dim=-1) + out.uncertainty
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


class TestDSDiscount:
    def _valid_belief_function(self, K=5, batch=4):
        belief = torch.rand(batch, K)
        belief = belief / (belief.sum(dim=-1, keepdim=True) + 1.0)  # leaves room for uncertainty
        uncertainty = 1.0 - belief.sum(dim=-1)
        return belief, uncertainty

    def test_identity_at_r_zero(self):
        belief, uncertainty = self._valid_belief_function()
        r = torch.zeros(belief.shape[0])
        b2, u2 = ds_discount(belief, uncertainty, r)
        assert torch.allclose(b2, belief, atol=1e-6)
        assert torch.allclose(u2, uncertainty, atol=1e-6)

    def test_total_ignorance_at_r_one(self):
        belief, uncertainty = self._valid_belief_function()
        r = torch.ones(belief.shape[0])
        b2, u2 = ds_discount(belief, uncertainty, r)
        assert torch.allclose(b2, torch.zeros_like(b2), atol=1e-6)
        assert torch.allclose(u2, torch.ones_like(u2), atol=1e-6)

    def test_mass_budget_preserved_for_all_r(self):
        belief, uncertainty = self._valid_belief_function()
        for r_val in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = torch.full((belief.shape[0],), r_val)
            b2, u2 = ds_discount(belief, uncertainty, r)
            total = b2.sum(dim=-1) + u2
            assert torch.allclose(total, torch.ones_like(total), atol=1e-5), f"failed at r={r_val}"


class TestKLAnnealing:
    def test_monotonic_and_bounded(self):
        vals = [kl_annealing_coefficient(e, annealing_steps=10) for e in range(0, 20)]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert vals == sorted(vals), "annealing coefficient must be non-decreasing"
        assert vals[-1] == 1.0, "must saturate at 1.0 after annealing_steps"

    def test_zero_annealing_steps_returns_one_immediately(self):
        assert kl_annealing_coefficient(0, annealing_steps=0) == 1.0

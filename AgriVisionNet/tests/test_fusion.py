"""
tests/test_fusion.py

Unit tests for models/fusion/task_hierarchy_fusion.py and the assembled
AgriVisionNet model, including the ablation configurations required by the
frozen architecture (see models/fusion/task_hierarchy_fusion.py docstring,
"ABLATIONS THIS DESIGN REQUIRES").

NOTE ON EXECUTION: requires torch + timm; not runnable in the authoring
sandbox (no network/GPU access there). Run via `pytest tests/` locally
after `pip install -r requirements.txt`.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed in this environment")
pytest.importorskip("timm", reason="timm not installed in this environment")

import yaml  # noqa: E402
from models.fusion.task_hierarchy_fusion import TaskHierarchyEvidentialFusion, PerTaskReliabilityGate  # noqa: E402
from models.agrivisionnet import build_model  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


class TestPerTaskReliabilityGate:
    def test_gates_differ_across_tasks(self):
        """The whole point of task-aware gating is that tasks CAN learn
        different gates -- verify the three gates are not tied to the
        same weights (a regression here would silently collapse the
        'task-aware' claim back to a shared gate)."""
        gate = PerTaskReliabilityGate(dim=32, hidden_dim=16)
        params = {t: list(gate.gates[t].parameters()) for t in ("crop", "disease", "severity")}
        # different modules -> different parameter tensors (by identity)
        assert params["crop"][0] is not params["disease"][0]
        assert params["disease"][0] is not params["severity"][0]

    def test_output_shape_and_range(self):
        gate = PerTaskReliabilityGate(dim=64, hidden_dim=16)
        r_cnn = torch.rand(8)
        r_swin = torch.rand(8)
        gates = gate(r_cnn, r_swin)
        for task in ("crop", "disease", "severity"):
            assert gates[task].shape == (8, 64)
            assert (gates[task] >= 0).all() and (gates[task] <= 1).all()


class TestTaskHierarchyEvidentialFusion:
    def test_forward_shapes(self):
        fusion = TaskHierarchyEvidentialFusion(
            cnn_channels=1280, swin_channels=768, projected_dim=512,
            num_disease_classes=38,
        )
        cnn_feat = torch.randn(4, 1280, 7, 7)
        swin_feat = torch.randn(4, 768, 7, 7)
        fused, aux = fusion(cnn_feat, swin_feat)

        for task in ("crop", "disease", "severity"):
            assert fused[task].shape == (4, 512)

        assert aux["aux_dirichlet_cnn"].uncertainty.shape == (4,)
        assert aux["aux_dirichlet_swin"].uncertainty.shape == (4,)

    def test_mismatched_spatial_resolution_is_handled(self):
        """Branches at different spatial resolutions must still fuse
        (Stage 0 bilinear-aligns them) -- guards against assuming both
        backbones always emit identical grid sizes."""
        fusion = TaskHierarchyEvidentialFusion(
            cnn_channels=1280, swin_channels=768, projected_dim=256,
            num_disease_classes=10,
        )
        cnn_feat = torch.randn(2, 1280, 7, 7)
        swin_feat = torch.randn(2, 768, 8, 8)  # deliberately mismatched
        fused, _ = fusion(cnn_feat, swin_feat)
        assert fused["disease"].shape == (2, 256)


class TestEndToEndModel:
    def test_full_forward_shapes(self, cfg):
        model = build_model(cfg)
        model.eval()
        x = torch.randn(2, 3, cfg["backbone"]["input_size"], cfg["backbone"]["input_size"])
        out = model(x, return_aux=True)

        assert out["crop_dirichlet"].uncertainty.shape == (2,)
        assert out["disease_dirichlet"].uncertainty.shape == (2,)
        assert out["severity_dirichlet"].uncertainty.shape == (2,)
        assert out["severity_uncertainty_discounted"].shape == (2,)
        assert out["fusion_aux"]["gate_values"]["disease"].shape[0] == 2

    def test_discounting_ablation_changes_severity_uncertainty(self, cfg):
        """with vs. without Stage 5 discounting must produce DIFFERENT
        severity uncertainty whenever disease uncertainty is non-trivial --
        otherwise the 'hierarchy-aware' ablation has nothing to show."""
        model = build_model(cfg)
        model.eval()
        x = torch.randn(4, 3, cfg["backbone"]["input_size"], cfg["backbone"]["input_size"])

        with torch.no_grad():
            out_discounted = model(x, discount_severity=True)
            out_plain = model(x, discount_severity=False)

        # they should differ unless disease uncertainty happened to be ~0
        # for every sample, which is astronomically unlikely on a random,
        # untrained forward pass
        assert not torch.allclose(
            out_discounted["severity_uncertainty_discounted"],
            out_plain["severity_uncertainty_discounted"],
        )

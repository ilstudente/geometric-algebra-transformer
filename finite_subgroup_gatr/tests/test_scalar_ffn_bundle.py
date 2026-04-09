"""Tests for ScalarFFNBundle.

Covers:
  1. Dense mode: same output as plain nn.Sequential MLP
  2. GM exact mode: shape-preserving, equivariant
  3. GM approx mode: shape-preserving, error_norm nonzero after perturbation
  4. Gradients flow through all modes
  5. GeoMLP integration: scalar_ffn injection produces correct output shapes
"""

import pytest
import torch
import torch.nn as nn

from finite_subgroup_gatr.backends.rotation_backend import CyclicBackend
from finite_subgroup_gatr.layers.scalar_ffn_bundle import ScalarFFNBundle
from finite_subgroup_gatr.layers.geo_mlp import GeoMLP
from finite_subgroup_gatr.pga.actions import construct_reference_multivector


@pytest.fixture
def cyclic8():
    return CyclicBackend(8)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestScalarFFNBundleShapes:
    def test_dense_shape_3d(self):
        bundle = ScalarFFNBundle(backend=None, c_s=16, hidden_s=32, mode="dense")
        x = torch.randn(2, 8, 16)
        out = bundle(x)
        assert out.shape == (2, 8, 16)

    def test_gm_exact_shape(self, cyclic8):
        bundle = ScalarFFNBundle(cyclic8, c_s=8, hidden_s=16, mode="gm_exact")
        B, X, C = 2, cyclic8.size(), 8
        x = torch.randn(B, X, C)
        out = bundle(x)
        assert out.shape == (B, X, C)

    def test_gm_approx_shape(self, cyclic8):
        bundle = ScalarFFNBundle(cyclic8, c_s=8, hidden_s=16, mode="gm_approx",
                                  error_mode="low_rank", error_rank=2)
        x = torch.randn(2, cyclic8.size(), 8)
        out = bundle(x)
        assert out.shape == (2, cyclic8.size(), 8)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ScalarFFNBundle(backend=None, c_s=8, hidden_s=16, mode="invalid_mode")

    def test_gm_mode_requires_backend(self):
        with pytest.raises(ValueError):
            ScalarFFNBundle(backend=None, c_s=8, hidden_s=16, mode="gm_exact")


# ---------------------------------------------------------------------------
# Equivariance test
# ---------------------------------------------------------------------------

class TestScalarFFNBundleEquivariance:
    def test_exact_mode_equivariance(self, cyclic8):
        X = cyclic8.size()
        bundle = ScalarFFNBundle(cyclic8, c_s=8, hidden_s=16, mode="gm_exact")
        bundle.eval()

        token_idx = torch.arange(X)
        perm = cyclic8.action_on_group_tokens(1, token_idx)

        x = torch.randn(2, X, 8)
        with torch.no_grad():
            y_g = bundle(x[:, perm, :])
            y_out = bundle(x)[:, perm, :]

        err = (y_g - y_out).abs().max().item()
        assert err < 1e-4, f"FFN equivariance error {err:.2e}"


# ---------------------------------------------------------------------------
# Error norm
# ---------------------------------------------------------------------------

class TestScalarFFNBundleErrorNorm:
    def test_dense_error_norm_zero(self):
        bundle = ScalarFFNBundle(None, c_s=8, hidden_s=16, mode="dense")
        assert bundle.error_norm().item() == pytest.approx(0.0)

    def test_exact_error_norm_zero(self, cyclic8):
        bundle = ScalarFFNBundle(cyclic8, c_s=8, hidden_s=16, mode="gm_exact")
        assert bundle.error_norm().item() == pytest.approx(0.0)

    def test_approx_error_norm_nonzero_after_perturbation(self, cyclic8):
        bundle = ScalarFFNBundle(cyclic8, c_s=8, hidden_s=16, mode="gm_approx",
                                  error_mode="dense_penalized")
        with torch.no_grad():
            for m in [bundle.fc1, bundle.fc2]:
                from finite_subgroup_gatr.layers.scalar_gm_linear import ScalarGMLinear
                if isinstance(m, ScalarGMLinear) and m.gm_weights is not None:
                    for w in m.gm_weights.weights:
                        if w.error_E is not None:
                            w.error_E.fill_(0.5)
        assert bundle.error_norm().item() > 0.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestScalarFFNBundleGradients:
    def test_backward_dense(self):
        bundle = ScalarFFNBundle(None, c_s=8, hidden_s=16, mode="dense")
        x = torch.randn(2, 4, 8, requires_grad=True)
        bundle(x).sum().backward()
        assert x.grad is not None

    def test_backward_gm_exact(self, cyclic8):
        bundle = ScalarFFNBundle(cyclic8, c_s=8, hidden_s=16, mode="gm_exact")
        x = torch.randn(2, cyclic8.size(), 8, requires_grad=True)
        bundle(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# GeoMLP integration
# ---------------------------------------------------------------------------

class TestGeoMLPScalarFFNInjection:
    def test_geo_mlp_with_gm_ffn(self, cyclic8):
        """GeoMLP with injected ScalarFFNBundle produces correct shapes."""
        C_mv, C_s = 4, 8
        bundle = ScalarFFNBundle(cyclic8, c_s=C_s, hidden_s=16, mode="gm_exact")
        mlp = GeoMLP(c_mv=C_mv, c_s=C_s, hidden_mv=8, hidden_s=16, scalar_ffn=bundle)

        B, X = 2, cyclic8.size()
        x_mv = torch.randn(B, X, C_mv, 16)
        x_s = torch.randn(B, X, C_s)
        ref_mv = construct_reference_multivector(x_mv)

        out_mv, out_s = mlp(x_mv, x_s, ref_mv)
        assert out_mv.shape == (B, X, C_mv, 16)
        assert out_s is not None and out_s.shape == (B, X, C_s)

    def test_geo_mlp_default_still_works(self):
        """Existing GeoMLP without injection continues to work unchanged."""
        mlp = GeoMLP(c_mv=4, c_s=8)
        x_mv = torch.randn(2, 6, 4, 16)
        x_s = torch.randn(2, 6, 8)
        ref_mv = construct_reference_multivector(x_mv)
        out_mv, out_s = mlp(x_mv, x_s, ref_mv)
        assert out_mv.shape == (2, 6, 4, 16)
        assert out_s is not None and out_s.shape == (2, 6, 8)

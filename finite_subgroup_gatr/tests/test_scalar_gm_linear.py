"""Tests for ScalarGMLinear.

Covers:
  1. Shape preservation for various leading-dim layouts
  2. Exact equivariance when error_mode="none"
  3. Error norm is zero for exact mode, nonzero after perturbation
  4. share_across_channels mode produces correct output shape
"""

import pytest
import torch

from finite_subgroup_gatr.backends.rotation_backend import CyclicBackend, OctahedralBackend
from finite_subgroup_gatr.layers.scalar_gm_linear import ScalarGMLinear


@pytest.fixture
def cyclic8():
    return CyclicBackend(8)


@pytest.fixture
def octahedral():
    return OctahedralBackend()


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestScalarGMLinearShapes:
    def test_basic_3d(self, cyclic8):
        """[B, X, C_in] → [B, X, C_out]"""
        layer = ScalarGMLinear(cyclic8, c_in=8, c_out=16, neighborhood_radius=1)
        B, X, C_in = 2, cyclic8.size(), 8
        x = torch.randn(B, X, C_in)
        out = layer(x)
        assert out.shape == (B, X, 16)

    def test_4d_with_object_axis(self, cyclic8):
        """[B, N_obj, X, C_in] → [B, N_obj, X, C_out]"""
        layer = ScalarGMLinear(cyclic8, c_in=8, c_out=12, neighborhood_radius=1)
        B, N, X, C_in = 2, 5, cyclic8.size(), 8
        x = torch.randn(B, N, X, C_in)
        out = layer(x)
        assert out.shape == (B, N, X, 12)

    def test_5d_with_time_and_object(self, cyclic8):
        """[B, T, N, X, C_in] → [B, T, N, X, C_out]"""
        layer = ScalarGMLinear(cyclic8, c_in=4, c_out=8, neighborhood_radius=1)
        B, T, N, X, C = 1, 3, 4, cyclic8.size(), 4
        x = torch.randn(B, T, N, X, C)
        out = layer(x)
        assert out.shape == (B, T, N, X, 8)

    def test_no_bias(self, cyclic8):
        layer = ScalarGMLinear(cyclic8, c_in=4, c_out=4, neighborhood_radius=1, bias=False)
        x = torch.randn(2, cyclic8.size(), 4)
        out = layer(x)
        assert out.shape == (2, cyclic8.size(), 4)
        assert layer.bias is None

    def test_shared_weights_shape(self, cyclic8):
        layer = ScalarGMLinear(
            cyclic8, c_in=8, c_out=8, neighborhood_radius=1,
            share_across_channels=True,
        )
        x = torch.randn(2, cyclic8.size(), 8)
        out = layer(x)
        assert out.shape == (2, cyclic8.size(), 8)

    def test_octahedral_shapes(self, octahedral):
        layer = ScalarGMLinear(octahedral, c_in=16, c_out=32, neighborhood_radius=1)
        B, X, C = 3, octahedral.size(), 16
        x = torch.randn(B, X, C)
        out = layer(x)
        assert out.shape == (B, X, 32)


# ---------------------------------------------------------------------------
# Equivariance tests
# ---------------------------------------------------------------------------

class TestScalarGMLinearEquivariance:
    """Exact equivariance: ScalarGMLinear(T_g x) == T_g ScalarGMLinear(x)."""

    def _check_equivariance(self, backend, c_in, c_out, radius, g_idx, tol=1e-4):
        X = backend.size()
        layer = ScalarGMLinear(backend, c_in=c_in, c_out=c_out,
                               neighborhood_radius=radius, error_mode="none")
        layer.eval()

        B = 2
        x = torch.randn(B, X, c_in)

        # Token permutation for g
        token_idx = torch.arange(X)
        perm = backend.action_on_group_tokens(g_idx, token_idx)  # [X]

        with torch.no_grad():
            # f(T_g x)
            x_g = x[:, perm, :]
            y_g = layer(x_g)

            # T_g f(x)
            y = layer(x)
            y_out = y[:, perm, :]

        err = (y_g - y_out).abs().max().item()
        assert err < tol, f"Equivariance error {err:.2e} > tol {tol:.2e} for g={g_idx}"

    def test_cyclic8_g1(self, cyclic8):
        self._check_equivariance(cyclic8, c_in=8, c_out=8, radius=1, g_idx=1)

    def test_cyclic8_g3_radius2(self, cyclic8):
        self._check_equivariance(cyclic8, c_in=4, c_out=16, radius=2, g_idx=3)

    def test_octahedral_g5(self, octahedral):
        self._check_equivariance(octahedral, c_in=8, c_out=8, radius=1, g_idx=5)

    def test_octahedral_different_c(self, octahedral):
        self._check_equivariance(octahedral, c_in=4, c_out=12, radius=1, g_idx=2)

    def test_4d_input_equivariance(self, cyclic8):
        """Equivariance holds with extra leading object axis."""
        X = cyclic8.size()
        layer = ScalarGMLinear(cyclic8, c_in=4, c_out=4, neighborhood_radius=1, error_mode="none")
        layer.eval()

        token_idx = torch.arange(X)
        perm = cyclic8.action_on_group_tokens(1, token_idx)

        x = torch.randn(2, 5, X, 4)  # [B, N_obj, X, C]
        with torch.no_grad():
            y_g = layer(x[:, :, perm, :])
            y_out = layer(x)[:, :, perm, :]

        err = (y_g - y_out).abs().max().item()
        assert err < 1e-4


# ---------------------------------------------------------------------------
# Error norm tests
# ---------------------------------------------------------------------------

class TestScalarGMLinearErrorNorm:
    def test_error_norm_zero_for_exact(self, cyclic8):
        layer = ScalarGMLinear(cyclic8, c_in=8, c_out=8,
                               neighborhood_radius=1, error_mode="none")
        assert layer.error_norm().item() == pytest.approx(0.0)

    def test_error_norm_nonzero_after_perturbation(self, cyclic8):
        layer = ScalarGMLinear(cyclic8, c_in=8, c_out=8,
                               neighborhood_radius=1, error_mode="dense_penalized")
        with torch.no_grad():
            for w in layer.gm_weights.weights:
                if w.error_E is not None:
                    w.error_E.fill_(1.0)
        assert layer.error_norm().item() > 0.0

    def test_low_rank_error_norm(self, cyclic8):
        layer = ScalarGMLinear(cyclic8, c_in=8, c_out=8,
                               neighborhood_radius=1, error_mode="low_rank", error_rank=2)
        with torch.no_grad():
            for w in layer.gm_weights.weights:
                if w.error_U is not None:
                    w.error_U.fill_(0.5)
                    w.error_V.fill_(0.5)
        assert layer.error_norm().item() > 0.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestScalarGMLinearGradients:
    def test_backward_exact(self, cyclic8):
        layer = ScalarGMLinear(cyclic8, c_in=8, c_out=8, neighborhood_radius=1)
        x = torch.randn(2, cyclic8.size(), 8, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_backward_approx(self, cyclic8):
        layer = ScalarGMLinear(cyclic8, c_in=4, c_out=4,
                               neighborhood_radius=1, error_mode="low_rank", error_rank=2)
        x = torch.randn(2, cyclic8.size(), 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        # ApproxLinear.bias inside GMMixingWeights is intentionally unused
        # (ScalarGMLinear calls effective_weight() directly, matching GMTokenMixer).
        # Check that all parameters EXCEPT those per-element dead biases have gradients.
        for name, p in layer.named_parameters():
            if "gm_weights" in name and name.endswith(".bias"):
                continue  # per-element ApproxLinear biases are never in the graph
            assert p.grad is not None, f"No grad for param {name}"

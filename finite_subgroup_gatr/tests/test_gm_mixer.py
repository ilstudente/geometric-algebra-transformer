"""Tests for GM token mixer.

Checks:
  - Exact equivariance when error_mode="none"
  - Equivariance error increases under controlled perturbation
  - Subgroup pooling shape correctness
"""

import pytest
import torch

from finite_subgroup_gatr.backends.rotation_backend import CyclicBackend, OctahedralBackend
from finite_subgroup_gatr.gm.token_mixer import GMTokenMixer
from finite_subgroup_gatr.gm.subgroup_pool import SubgroupPool


@pytest.fixture
def cyclic8():
    return CyclicBackend(8)


@pytest.fixture
def octahedral():
    return OctahedralBackend()


class TestGMTokenMixerEquivariance:
    """Test exact equivariance of GM mixer."""

    def _check_equivariance(self, backend, c_mv, c_s, radius, g_idx, B=2, tol=1e-4):
        """Check that GMConv(T_g x) = T_g GMConv(x)."""
        X = backend.size()
        mixer = GMTokenMixer(backend, c_mv, c_s, radius, error_mode="none")
        mixer.eval()

        x_mv = torch.randn(B, X, c_mv, 16)
        x_s = torch.randn(B, X, c_s) if c_s is not None else None

        with torch.no_grad():
            # Apply group action to input
            perm = backend.left_action_table[g_idx]  # [X]: perm[x] = g*x
            x_mv_g = x_mv[:, perm, :, :]
            x_s_g = x_s[:, perm, :] if x_s is not None else None

            # Also apply MV rotation
            M_g = backend.mv_rep[g_idx].float()
            x_mv_g = x_mv_g @ M_g.T

            # Forward on transformed input
            y_mv_g, y_s_g = mixer(x_mv_g, x_s_g)

            # Forward on original, then transform
            y_mv, y_s = mixer(x_mv, x_s)
            y_mv_out = y_mv[:, perm, :, :] @ M_g.T
            y_s_out = y_s[:, perm, :] if y_s is not None else None

        mv_err = (y_mv_g - y_mv_out).abs().max().item()
        assert mv_err < tol, f"GM equivariance error too large: {mv_err:.2e}"

        if y_s_g is not None:
            s_err = (y_s_g - y_s_out).abs().max().item()
            assert s_err < tol, f"Scalar equivariance error too large: {s_err:.2e}"

    def test_cyclic8_g1_exact(self, cyclic8):
        self._check_equivariance(cyclic8, c_mv=4, c_s=8, radius=1, g_idx=1)

    def test_cyclic8_g3_exact(self, cyclic8):
        self._check_equivariance(cyclic8, c_mv=4, c_s=None, radius=2, g_idx=3)

    def test_octahedral_g5_exact(self, octahedral):
        self._check_equivariance(octahedral, c_mv=4, c_s=8, radius=1, g_idx=5)


class TestGMTokenMixerApprox:
    """Test that approximate equivariance error increases under larger perturbation."""

    def _measure_equiv_error(self, backend, error_scale, c_mv=4, c_s=None, radius=1):
        X = backend.size()
        mixer = GMTokenMixer(backend, c_mv, c_s, radius, error_mode="dense_penalized")

        # Set the error to a specific scale
        with torch.no_grad():
            for layer in mixer.mv_weights.weights:
                if layer.error_E is not None:
                    layer.error_E.fill_(0.0)
                    layer.error_E.add_(torch.randn_like(layer.error_E) * error_scale)

        mixer.eval()
        B = 2
        x_mv = torch.randn(B, X, c_mv, 16)

        errors = []
        for g_idx in range(1, 4):
            perm = backend.left_action_table[g_idx]
            M_g = backend.mv_rep[g_idx].float()
            x_mv_g = (x_mv[:, perm, :, :] @ M_g.T)
            with torch.no_grad():
                y_mv_g, _ = mixer(x_mv_g, None)
                y_mv, _ = mixer(x_mv, None)
                y_mv_out = y_mv[:, perm, :, :] @ M_g.T
            err = (y_mv_g - y_mv_out).pow(2).mean().item()
            errors.append(err)

        return sum(errors) / len(errors)

    def test_error_increases_with_perturbation(self, cyclic8):
        err_small = self._measure_equiv_error(cyclic8, error_scale=0.01)
        err_large = self._measure_equiv_error(cyclic8, error_scale=1.0)
        assert err_large > err_small, \
            f"Expected larger error with bigger perturbation: {err_small:.4f} vs {err_large:.4f}"


class TestSubgroupPool:
    def test_output_shape(self):
        big = CyclicBackend(8)
        sub = big.subgroup("Z4")
        pool = SubgroupPool(big, sub)

        B, X = 3, big.size()
        x_mv = torch.randn(B, X, 4, 16)
        x_s = torch.randn(B, X, 8)

        out_mv, out_s = pool(x_mv, x_s)
        expected_cosets = big.size() // sub.size()
        assert out_mv.shape == (B, expected_cosets, 4, 16)
        assert out_s.shape == (B, expected_cosets, 8)

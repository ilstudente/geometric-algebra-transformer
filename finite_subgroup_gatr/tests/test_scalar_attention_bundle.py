"""Tests for ScalarAttentionProjectionBundle.

Covers:
  1. Shape: Q/K projections produce correct output dimensions
  2. Equivariance of Q/K projections in exact mode
  3. DiscreteMVAttention integration: injecting bundle preserves output shapes
  4. Bundle with project_values and project_output flags
"""

import pytest
import torch

from finite_subgroup_gatr.backends.rotation_backend import CyclicBackend
from finite_subgroup_gatr.layers.scalar_attention_bundle import ScalarAttentionProjectionBundle
from finite_subgroup_gatr.layers.discrete_attention import DiscreteMVAttention


@pytest.fixture
def cyclic8():
    return CyclicBackend(8)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestScalarAttentionBundleShapes:
    def test_qk_output_shape_dense(self):
        bundle = ScalarAttentionProjectionBundle(
            backend=None, c_s=16, num_heads=4, head_dim_s=4, mode="dense"
        )
        B, X, C_s = 2, 8, 16
        s = torch.randn(B, X, C_s)
        q = bundle.forward_q(s)
        k = bundle.forward_k(s)
        assert q.shape == (B, X, 4 * 4)
        assert k.shape == (B, X, 4 * 4)

    def test_qk_output_shape_gm(self, cyclic8):
        bundle = ScalarAttentionProjectionBundle(
            backend=cyclic8, c_s=8, num_heads=2, head_dim_s=4, mode="gm_exact"
        )
        B, X = 2, cyclic8.size()
        s = torch.randn(B, X, 8)
        q = bundle.forward_q(s)
        k = bundle.forward_k(s)
        assert q.shape == (B, X, 2 * 4)
        assert k.shape == (B, X, 2 * 4)

    def test_value_proj_none_by_default(self, cyclic8):
        bundle = ScalarAttentionProjectionBundle(
            backend=cyclic8, c_s=8, num_heads=2, head_dim_s=4, mode="gm_exact"
        )
        s = torch.randn(2, cyclic8.size(), 8)
        assert bundle.forward_v(s) is None

    def test_value_proj_active(self, cyclic8):
        bundle = ScalarAttentionProjectionBundle(
            backend=cyclic8, c_s=8, num_heads=2, head_dim_s=4,
            mode="gm_exact", project_values=True
        )
        s = torch.randn(2, cyclic8.size(), 8)
        v = bundle.forward_v(s)
        assert v is not None and v.shape == (2, cyclic8.size(), 8)

    def test_requires_backend_for_gm(self):
        with pytest.raises(ValueError):
            ScalarAttentionProjectionBundle(
                backend=None, c_s=8, num_heads=2, head_dim_s=4, mode="gm_exact"
            )


# ---------------------------------------------------------------------------
# Equivariance of Q/K projections
# ---------------------------------------------------------------------------

class TestScalarAttentionBundleEquivariance:
    def _check_proj_equivariance(self, backend, proj_fn, tol=1e-4):
        X = backend.size()
        token_idx = torch.arange(X)
        perm = backend.action_on_group_tokens(1, token_idx)

        s = torch.randn(2, X, 8)
        with torch.no_grad():
            out_g = proj_fn(s[:, perm, :])
            out_out = proj_fn(s)[:, perm, :]

        err = (out_g - out_out).abs().max().item()
        assert err < tol, f"Q/K equivariance error {err:.2e}"

    def test_q_proj_equivariant(self, cyclic8):
        bundle = ScalarAttentionProjectionBundle(
            cyclic8, c_s=8, num_heads=2, head_dim_s=4, mode="gm_exact"
        )
        bundle.eval()
        self._check_proj_equivariance(cyclic8, bundle.forward_q)

    def test_k_proj_equivariant(self, cyclic8):
        bundle = ScalarAttentionProjectionBundle(
            cyclic8, c_s=8, num_heads=2, head_dim_s=4, mode="gm_exact"
        )
        bundle.eval()
        self._check_proj_equivariance(cyclic8, bundle.forward_k)


# ---------------------------------------------------------------------------
# DiscreteMVAttention integration
# ---------------------------------------------------------------------------

class TestDiscreteMVAttentionWithBundle:
    def test_output_shapes_with_bundle(self, cyclic8):
        C_mv, C_s, H = 4, 8, 2
        head_dim_s = max(1, C_s // H)
        bundle = ScalarAttentionProjectionBundle(
            cyclic8, c_s=C_s, num_heads=H, head_dim_s=head_dim_s, mode="gm_exact"
        )
        attn = DiscreteMVAttention(
            c_mv=C_mv, c_s=C_s, num_heads=H,
            use_scalar_logits=True,
            scalar_attn_bundle=bundle,
        )
        B, X = 2, cyclic8.size()
        x_mv = torch.randn(B, X, C_mv, 16)
        x_s = torch.randn(B, X, C_s)
        out_mv, out_s = attn(x_mv, x_s)
        assert out_mv.shape == (B, X, C_mv, 16)
        assert out_s is not None and out_s.shape == (B, X, C_s)

    def test_default_attention_unchanged(self):
        """DiscreteMVAttention without bundle continues to work."""
        C_mv, C_s = 4, 8
        attn = DiscreteMVAttention(c_mv=C_mv, c_s=C_s, num_heads=2, use_scalar_logits=True)
        B, X = 2, 6
        x_mv = torch.randn(B, X, C_mv, 16)
        x_s = torch.randn(B, X, C_s)
        out_mv, out_s = attn(x_mv, x_s)
        assert out_mv.shape == (B, X, C_mv, 16)
        assert out_s is not None and out_s.shape == (B, X, C_s)

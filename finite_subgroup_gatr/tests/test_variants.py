"""End-to-end tests for all three variants."""

import pytest
import torch

from finite_subgroup_gatr.backends.rotation_backend import CyclicBackend, OctahedralBackend
from finite_subgroup_gatr.models.variant_a_gm_only import VariantA
from finite_subgroup_gatr.models.variant_b_gm_attention import VariantB
from finite_subgroup_gatr.models.variant_c_axial import VariantC


@pytest.fixture
def cyclic4():
    return CyclicBackend(4)


@pytest.fixture
def octahedral():
    return OctahedralBackend()


class TestVariantAShapes:
    def test_forward_shape(self, cyclic4):
        model = VariantA(
            backend=cyclic4,
            in_mv_channels=1,
            out_mv_channels=1,
            c_mv=4,
            in_s_channels=4,
            out_s_channels=2,
            c_s=8,
            num_blocks=2,
            neighborhood_radius=1,
        )
        B, X = 2, cyclic4.size()
        x_mv = torch.randn(B, X, 1, 16)
        x_s = torch.randn(B, X, 4)
        out_mv, out_s = model(x_mv, x_s)
        assert out_mv.shape == (B, X, 1, 16)
        assert out_s is not None and out_s.shape == (B, X, 2)

    def test_forward_no_scalars(self, cyclic4):
        model = VariantA(
            backend=cyclic4,
            in_mv_channels=1,
            out_mv_channels=2,
            c_mv=4,
            c_s=None,
            num_blocks=2,
        )
        B, X = 3, cyclic4.size()
        x_mv = torch.randn(B, X, 1, 16)
        out_mv, out_s = model(x_mv, None)
        assert out_mv.shape == (B, X, 2, 16)
        assert out_s is None

    def test_backward(self, cyclic4):
        model = VariantA(
            backend=cyclic4, in_mv_channels=1, out_mv_channels=1,
            c_mv=4, c_s=None, num_blocks=2,
        )
        x_mv = torch.randn(2, cyclic4.size(), 1, 16, requires_grad=True)
        out_mv, _ = model(x_mv, None)
        out_mv.sum().backward()
        assert x_mv.grad is not None


class TestVariantBShapes:
    def test_forward_shape(self, cyclic4):
        model = VariantB(
            backend=cyclic4,
            in_mv_channels=1,
            out_mv_channels=1,
            c_mv=4,
            c_s=8,
            num_blocks=2,
            num_heads=2,
        )
        B, X = 2, cyclic4.size()
        x_mv = torch.randn(B, X, 1, 16)
        x_s = torch.randn(B, X, 4)
        out_mv, out_s = model(x_mv, x_s)
        assert out_mv.shape == (B, X, 1, 16)


class TestVariantCShapes:
    def test_forward_shape(self, cyclic4):
        model = VariantC(
            backend=cyclic4,
            in_mv_channels=1,
            out_mv_channels=1,
            c_mv=4,
            c_s=8,
            num_cycles=1,
            num_heads=2,
        )
        B, N_obj, N_time, X = 2, 3, 4, cyclic4.size()
        x_mv = torch.randn(B, N_obj, N_time, X, 1, 16)
        x_s = torch.randn(B, N_obj, N_time, X, 4)
        out_mv, out_s = model(x_mv, x_s)
        assert out_mv.shape == (B, N_obj, N_time, X, 1, 16)


class TestApproxEquivariance:
    def test_exact_gm_has_lower_error(self, cyclic4):
        """With exact GM, equivariance error should be very small."""
        model = VariantA(
            backend=cyclic4, in_mv_channels=1, out_mv_channels=1,
            c_mv=4, c_s=None, num_blocks=2, error_mode="none",
        )
        model.eval()
        B, X = 2, cyclic4.size()
        x_mv = torch.randn(B, X, 1, 16)
        with torch.no_grad():
            err = model.equivariance_error(x_mv, num_samples=cyclic4.size())
        # Error should be very small for exact equivariant ops
        assert err.item() < 0.1, f"Equivariance error too large: {err.item():.4f}"

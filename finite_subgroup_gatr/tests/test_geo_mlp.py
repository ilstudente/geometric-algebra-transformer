"""Tests for GeoMLP."""

import pytest
import torch

from finite_subgroup_gatr.layers.geo_mlp import GeoMLP
from finite_subgroup_gatr.pga.actions import construct_reference_multivector


class TestGeoMLPShapes:
    def test_output_shapes(self):
        mlp = GeoMLP(c_mv=8, c_s=32, hidden_mv=16, hidden_s=64)
        B, X, C_mv, C_s = 2, 5, 8, 32
        x_mv = torch.randn(B, X, C_mv, 16)
        x_s = torch.randn(B, X, C_s)
        ref_mv = construct_reference_multivector(x_mv)

        out_mv, out_s = mlp(x_mv, x_s, ref_mv)
        assert out_mv.shape == (B, X, C_mv, 16)
        assert out_s is not None and out_s.shape == (B, X, C_s)

    def test_no_scalars(self):
        mlp = GeoMLP(c_mv=8, c_s=None)
        x_mv = torch.randn(3, 10, 8, 16)
        ref_mv = construct_reference_multivector(x_mv)
        out_mv, out_s = mlp(x_mv, None, ref_mv)
        assert out_mv.shape == (3, 10, 8, 16)
        assert out_s is None

    def test_reference_broadcast(self):
        """Reference multivector should broadcast correctly over channels."""
        mlp = GeoMLP(c_mv=4, c_s=None)
        x_mv = torch.randn(2, 6, 4, 16)
        # Reference with shape [2, 1, 1, 16] should broadcast
        ref_mv = torch.randn(2, 1, 1, 16)
        out_mv, _ = mlp(x_mv, None, ref_mv)
        assert out_mv.shape == (2, 6, 4, 16)


class TestGeoMLPGradients:
    def test_backward_passes(self):
        mlp = GeoMLP(c_mv=4, c_s=8)
        x_mv = torch.randn(2, 4, 4, 16, requires_grad=True)
        x_s = torch.randn(2, 4, 8, requires_grad=True)
        ref_mv = construct_reference_multivector(x_mv.detach())
        out_mv, out_s = mlp(x_mv, x_s, ref_mv)
        loss = out_mv.sum() + out_s.sum()
        loss.backward()
        assert x_mv.grad is not None
        assert x_s.grad is not None


class TestGeoMLPGeometricBilinear:
    def test_gp_output_shape(self):
        """GP branch should produce correct shape."""
        from gatr.primitives.bilinear import geometric_product
        x = torch.randn(3, 16)
        y = torch.randn(3, 16)
        out = geometric_product(x, y)
        assert out.shape == (3, 16)

    def test_join_output_shape(self):
        """Join branch should produce correct shape."""
        from gatr.primitives.dual import equivariant_join
        x = torch.randn(3, 4, 16)
        y = torch.randn(3, 4, 16)
        ref = torch.randn(3, 1, 16)
        out = equivariant_join(x, y, ref)
        assert out.shape == (3, 4, 16)

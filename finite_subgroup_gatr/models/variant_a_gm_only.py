"""Variant A: GM-only token mixer.

No attention. Token interaction solely via discrete-group convolution.
This is the purest finite-subgroup rebuild and the cleanest baseline.

Input:  [B, X, in_mv_channels, 16] + optional [B, X, in_s_channels]
Output: [B, X, out_mv_channels, 16] + optional [B, X, out_s_channels]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gatr.layers.linear import EquiLinear
from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.layers.common_blocks import GMOnlyBlock
from finite_subgroup_gatr.layers.geo_mlp import GeoMLP
from finite_subgroup_gatr.gm.token_mixer import GMTokenMixer
from finite_subgroup_gatr.pga.actions import construct_reference_multivector


class VariantA(nn.Module):
    """FiniteSubgroupGATr Variant A: GM-only token mixer.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    in_mv_channels : int
    out_mv_channels : int
    c_mv : int
        Hidden multivector channels.
    in_s_channels : int or None
    out_s_channels : int or None
    c_s : int or None
        Hidden scalar channels.
    num_blocks : int
    neighborhood_radius : int
    error_mode : {"none", "low_rank", "dense_penalized"}
    error_rank : int
    """

    def __init__(
        self,
        backend: FiniteSymmetryBackend,
        in_mv_channels: int,
        out_mv_channels: int,
        c_mv: int = 8,
        in_s_channels: int | None = None,
        out_s_channels: int | None = None,
        c_s: int | None = 32,
        num_blocks: int = 6,
        neighborhood_radius: int = 1,
        error_mode: str = "none",
        error_rank: int = 4,
        # Scalar-path GM options (Variant 3 / ScalarPathGM)
        scalar_mlp_mode: str = "dense",
        scalar_mlp_radius: int = 1,
        scalar_mlp_error_mode: str = "none",
        scalar_mlp_error_rank: int = 4,
    ):
        super().__init__()
        self.backend = backend
        self.c_mv = c_mv
        self.c_s = c_s

        # Input embedding
        self.embed_in = EquiLinear(
            in_mv_channels, c_mv,
            in_s_channels=in_s_channels,
            out_s_channels=c_s,
        )

        # Backbone blocks
        self.blocks = nn.ModuleList([
            GMOnlyBlock(
                c_mv=c_mv,
                c_s=c_s,
                backend=backend,
                neighborhood_radius=neighborhood_radius,
                error_mode=error_mode,
                error_rank=error_rank,
                scalar_mlp_mode=scalar_mlp_mode,
                scalar_mlp_radius=scalar_mlp_radius,
                scalar_mlp_error_mode=scalar_mlp_error_mode,
                scalar_mlp_error_rank=scalar_mlp_error_rank,
            )
            for _ in range(num_blocks)
        ])

        # Output projection
        self.embed_out = EquiLinear(
            c_mv, out_mv_channels,
            in_s_channels=c_s,
            out_s_channels=out_s_channels,
        )

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_mv : Tensor [B, X, in_mv_channels, 16]
        x_s  : Tensor [B, X, in_s_channels] or None

        Returns
        -------
        out_mv : Tensor [B, X, out_mv_channels, 16]
        out_s  : Tensor [B, X, out_s_channels] or None
        """
        # Input embedding: operates per token
        # EquiLinear expects [..., C, 16] so we pass [B*X, C, 16]
        B, X = x_mv.shape[:2]
        h_mv = x_mv.reshape(B * X, *x_mv.shape[2:])
        h_s = x_s.reshape(B * X, *x_s.shape[2:]) if x_s is not None else None
        h_mv, h_s = self.embed_in(h_mv, scalars=h_s)
        h_mv = h_mv.reshape(B, X, *h_mv.shape[1:])
        if h_s is not None:
            h_s = h_s.reshape(B, X, *h_s.shape[1:])

        # Backbone
        for block in self.blocks:
            h_mv, h_s = block(h_mv, h_s)

        # Output projection
        h_mv_flat = h_mv.reshape(B * X, *h_mv.shape[2:])
        h_s_flat = h_s.reshape(B * X, *h_s.shape[2:]) if h_s is not None else None
        out_mv, out_s = self.embed_out(h_mv_flat, scalars=h_s_flat)
        out_mv = out_mv.reshape(B, X, *out_mv.shape[1:])
        if out_s is not None:
            out_s = out_s.reshape(B, X, *out_s.shape[1:])

        return out_mv, out_s

    def equivariance_error(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
        num_samples: int = 8,
    ) -> Tensor:
        """Measure finite-group equivariance error."""
        from finite_subgroup_gatr.pga.actions import equivariance_error
        def model_fn(mv, s):
            return self(mv, s)
        return equivariance_error(self.backend, model_fn, x_mv, x_s, num_samples)

    def error_norm(self) -> Tensor:
        """Total residual error norm across all GM mixer blocks."""
        total = torch.zeros(1)
        for block in self.blocks:
            if hasattr(block, 'token_mixer'):
                total = total + block.token_mixer.error_norm().cpu()
        return total

"""Variant B: GM + discrete multivector attention.

Most expressive variant. Each block:
  MVNorm -> GMTokenMixer -> +
  MVNorm -> DiscreteMVAttention -> +
  MVNorm -> GeoMLP -> +
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gatr.layers.linear import EquiLinear
from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.layers.common_blocks import GMAttentionBlock


class VariantB(nn.Module):
    """FiniteSubgroupGATr Variant B: GM + discrete attention.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    in_mv_channels : int
    out_mv_channels : int
    c_mv : int
    in_s_channels : int or None
    out_s_channels : int or None
    c_s : int or None
    num_blocks : int
    neighborhood_radius : int
    num_heads : int
    error_mode : str
    error_rank : int
    use_scalar_logits : bool
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
        num_heads: int = 4,
        error_mode: str = "none",
        error_rank: int = 4,
        use_scalar_logits: bool = True,
    ):
        super().__init__()
        self.backend = backend
        self.c_mv = c_mv
        self.c_s = c_s

        self.embed_in = EquiLinear(
            in_mv_channels, c_mv,
            in_s_channels=in_s_channels,
            out_s_channels=c_s,
        )

        self.blocks = nn.ModuleList([
            GMAttentionBlock(
                c_mv=c_mv,
                c_s=c_s,
                backend=backend,
                neighborhood_radius=neighborhood_radius,
                num_heads=num_heads,
                error_mode=error_mode,
                error_rank=error_rank,
                use_scalar_logits=use_scalar_logits,
            )
            for _ in range(num_blocks)
        ])

        self.embed_out = EquiLinear(
            c_mv, out_mv_channels,
            in_s_channels=c_s,
            out_s_channels=out_s_channels,
        )

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_mv : Tensor [B, X, in_mv_channels, 16]
        x_s  : Tensor [B, X, in_s_channels] or None
        attn_mask : Tensor [B, X, X] or None

        Returns
        -------
        out_mv : Tensor [B, X, out_mv_channels, 16]
        out_s  : Tensor [B, X, out_s_channels] or None
        """
        B, X = x_mv.shape[:2]
        h_mv = x_mv.reshape(B * X, *x_mv.shape[2:])
        h_s = x_s.reshape(B * X, *x_s.shape[2:]) if x_s is not None else None
        h_mv, h_s = self.embed_in(h_mv, scalars=h_s)
        h_mv = h_mv.reshape(B, X, *h_mv.shape[1:])
        if h_s is not None:
            h_s = h_s.reshape(B, X, *h_s.shape[1:])

        for block in self.blocks:
            h_mv, h_s = block(h_mv, h_s, attn_mask=attn_mask)

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
        from finite_subgroup_gatr.pga.actions import equivariance_error
        def model_fn(mv, s):
            return self(mv, s)
        return equivariance_error(self.backend, model_fn, x_mv, x_s, num_samples)

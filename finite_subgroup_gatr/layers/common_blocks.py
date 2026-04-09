"""Common block building blocks shared across all three variants.

FiniteSubgroupBlock: the outer skeleton:
  MVNorm -> mixer -> + -> MVNorm -> GeoMLP -> +

where the mixer is injected as a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from gatr.primitives.normalization import equi_layer_norm
from finite_subgroup_gatr.layers.geo_mlp import GeoMLP
from finite_subgroup_gatr.pga.actions import construct_reference_multivector


class FiniteSubgroupBlock(nn.Module):
    """Residual block with a pluggable token mixer and shared GeoMLP.

    Structure:
      input
        -> MVNorm -> token_mixer -> + residual (pre-mixer skip)
        -> MVNorm -> GeoMLP -> + residual (post-MLP skip)
      output

    Parameters
    ----------
    c_mv : int
    c_s : int or None
    token_mixer : nn.Module
        Any module with signature (x_mv, x_s) -> (x_mv, x_s).
    geo_mlp : GeoMLP
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        token_mixer: nn.Module,
        geo_mlp: GeoMLP,
    ):
        super().__init__()
        self.c_mv = c_mv
        self.c_s = c_s
        self.token_mixer = token_mixer
        self.geo_mlp = geo_mlp

        if c_s is not None:
            self.norm1_s = nn.LayerNorm(c_s)
            self.norm2_s = nn.LayerNorm(c_s)
        else:
            self.norm1_s = None
            self.norm2_s = None

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None,
        ref_mv: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_mv : Tensor [..., X, C_mv, 16]
        x_s  : Tensor [..., X, C_s] or None
        ref_mv : Tensor [..., 1, 16] or None
            Reference multivector for GeoMLP join. If None, computed from x_mv.

        Returns
        -------
        out_mv : Tensor [..., X, C_mv, 16]
        out_s  : Tensor [..., X, C_s] or None
        """
        # Token mixing sub-block
        h_mv = equi_layer_norm(x_mv)
        h_s = self.norm1_s(x_s) if (x_s is not None and self.norm1_s is not None) else x_s
        h_mv, h_s = self.token_mixer(h_mv, h_s)
        x_mv = x_mv + h_mv
        if x_s is not None and h_s is not None:
            x_s = x_s + h_s

        # GeoMLP sub-block
        h_mv = equi_layer_norm(x_mv)
        h_s = self.norm2_s(x_s) if (x_s is not None and self.norm2_s is not None) else x_s

        if ref_mv is None:
            ref_mv = construct_reference_multivector(h_mv)

        mlp_mv, mlp_s = self.geo_mlp(h_mv, h_s, ref_mv)
        x_mv = x_mv + mlp_mv
        if x_s is not None and mlp_s is not None:
            x_s = x_s + mlp_s

        return x_mv, x_s


class GMOnlyBlock(FiniteSubgroupBlock):
    """Block variant A: GM-only token mixer + GeoMLP."""

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        backend,
        neighborhood_radius: int,
        geo_mlp_hidden_mv: int | None = None,
        geo_mlp_hidden_s: int | None = None,
        error_mode: str = "none",
        error_rank: int = 4,
        # Scalar-path GM options (Variant 3 / ScalarPathGM)
        scalar_mlp_mode: str = "dense",
        scalar_mlp_radius: int = 1,
        scalar_mlp_error_mode: str = "none",
        scalar_mlp_error_rank: int = 4,
    ):
        from finite_subgroup_gatr.gm.token_mixer import GMTokenMixer
        mixer = GMTokenMixer(
            backend,
            c_mv,
            c_s,
            neighborhood_radius,
            error_mode,
            error_rank,
        )

        # Build scalar FFN bundle if scalar path is active and GM mode requested
        scalar_ffn = None
        if c_s is not None and scalar_mlp_mode != "dense":
            from finite_subgroup_gatr.layers.scalar_ffn_bundle import ScalarFFNBundle
            hidden_s = geo_mlp_hidden_s if geo_mlp_hidden_s is not None else 2 * c_s
            scalar_ffn = ScalarFFNBundle(
                backend=backend,
                c_s=c_s,
                hidden_s=hidden_s,
                mode=scalar_mlp_mode,
                neighborhood_radius=scalar_mlp_radius,
                error_mode=scalar_mlp_error_mode,
                error_rank=scalar_mlp_error_rank,
            )

        mlp = GeoMLP(
            c_mv,
            c_s,
            hidden_mv=geo_mlp_hidden_mv,
            hidden_s=geo_mlp_hidden_s,
            scalar_ffn=scalar_ffn,
        )
        super().__init__(c_mv, c_s, mixer, mlp)


class GMAttentionBlock(nn.Module):
    """Block variant B: GM token mixer + discrete MV attention + GeoMLP.

    Structure:
      input
        -> MVNorm -> GMTokenMixer -> + residual
        -> MVNorm -> DiscreteMVAttention -> + residual
        -> MVNorm -> GeoMLP -> + residual
      output
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        backend,
        neighborhood_radius: int,
        num_heads: int = 4,
        geo_mlp_hidden_mv: int | None = None,
        geo_mlp_hidden_s: int | None = None,
        error_mode: str = "none",
        error_rank: int = 4,
        use_scalar_logits: bool = True,
    ):
        super().__init__()
        from finite_subgroup_gatr.gm.token_mixer import GMTokenMixer
        from finite_subgroup_gatr.layers.discrete_attention import DiscreteMVAttention

        self.gm_mixer = GMTokenMixer(
            backend, c_mv, c_s, neighborhood_radius, error_mode, error_rank,
        )
        self.attention = DiscreteMVAttention(
            c_mv, c_s, num_heads=num_heads, use_scalar_logits=use_scalar_logits,
        )
        self.geo_mlp = GeoMLP(
            c_mv, c_s, hidden_mv=geo_mlp_hidden_mv, hidden_s=geo_mlp_hidden_s,
        )

        if c_s is not None:
            self.norm1_s = nn.LayerNorm(c_s)
            self.norm2_s = nn.LayerNorm(c_s)
            self.norm3_s = nn.LayerNorm(c_s)
        else:
            self.norm1_s = self.norm2_s = self.norm3_s = None

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None,
        ref_mv: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        # GM sub-block
        h_mv = equi_layer_norm(x_mv)
        h_s = self.norm1_s(x_s) if (x_s is not None and self.norm1_s is not None) else x_s
        h_mv, h_s = self.gm_mixer(h_mv, h_s)
        x_mv = x_mv + h_mv
        if x_s is not None and h_s is not None:
            x_s = x_s + h_s

        # Attention sub-block
        h_mv = equi_layer_norm(x_mv)
        h_s = self.norm2_s(x_s) if (x_s is not None and self.norm2_s is not None) else x_s
        h_mv, h_s = self.attention(h_mv, h_s, mask=attn_mask)
        x_mv = x_mv + h_mv
        if x_s is not None and h_s is not None:
            x_s = x_s + h_s

        # GeoMLP sub-block
        h_mv = equi_layer_norm(x_mv)
        h_s = self.norm3_s(x_s) if (x_s is not None and self.norm3_s is not None) else x_s
        if ref_mv is None:
            ref_mv = construct_reference_multivector(h_mv)
        mlp_mv, mlp_s = self.geo_mlp(h_mv, h_s, ref_mv)
        x_mv = x_mv + mlp_mv
        if x_s is not None and mlp_s is not None:
            x_s = x_s + mlp_s

        return x_mv, x_s

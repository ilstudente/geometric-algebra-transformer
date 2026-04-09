"""Variant C: Axial-split model.

GM mixing over the symmetry axis (X), attention over object and time axes.

Expected input: [B, N_obj, N_time, X, in_mv_channels, 16]

4-block cycle:
  1. SymmetryAxisGMBlock  (over X)
  2. ObjectAxisMVAttentionBlock  (over N_obj)
  3. SymmetryAxisGMBlock  (over X)
  4. TimeAxisMVAttentionBlock  (over N_time)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gatr.layers.linear import EquiLinear
from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.layers.axial_blocks import (
    SymmetryAxisGMBlock,
    ObjectAxisMVAttentionBlock,
    TimeAxisMVAttentionBlock,
)


class VariantC(nn.Module):
    """FiniteSubgroupGATr Variant C: Axial split.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    in_mv_channels : int
    out_mv_channels : int
    c_mv : int
    in_s_channels : int or None
    out_s_channels : int or None
    c_s : int or None
    num_cycles : int
        Number of 4-block cycles.
    neighborhood_radius : int
    num_heads : int
    error_mode : str
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
        num_cycles: int = 3,
        neighborhood_radius: int = 1,
        num_heads: int = 4,
        error_mode: str = "none",
        error_rank: int = 4,
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

        self.gm_blocks = nn.ModuleList([
            SymmetryAxisGMBlock(
                c_mv=c_mv,
                c_s=c_s,
                backend=backend,
                neighborhood_radius=neighborhood_radius,
                error_mode=error_mode,
                error_rank=error_rank,
            )
            for _ in range(2 * num_cycles)  # 2 GM blocks per cycle
        ])

        self.obj_attn_blocks = nn.ModuleList([
            ObjectAxisMVAttentionBlock(c_mv=c_mv, c_s=c_s, num_heads=num_heads)
            for _ in range(num_cycles)
        ])

        self.time_attn_blocks = nn.ModuleList([
            TimeAxisMVAttentionBlock(c_mv=c_mv, c_s=c_s, num_heads=num_heads)
            for _ in range(num_cycles)
        ])

        self.embed_out = EquiLinear(
            c_mv, out_mv_channels,
            in_s_channels=c_s,
            out_s_channels=out_s_channels,
        )

        self.num_cycles = num_cycles

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_mv : Tensor [B, N_obj, N_time, X, in_mv_channels, 16]
        x_s  : Tensor [B, N_obj, N_time, X, in_s_channels] or None

        Returns
        -------
        out_mv : Tensor [B, N_obj, N_time, X, out_mv_channels, 16]
        out_s  : Tensor [B, N_obj, N_time, X, out_s_channels] or None
        """
        # [B, N_obj, N_time, X, C, 16]
        shape = x_mv.shape
        B, N_obj, N_time, X = shape[:4]
        C_in = shape[4]

        # Embed: flatten all leading dims
        flat = x_mv.reshape(-1, C_in, 16)
        flat_s = x_s.reshape(-1, x_s.shape[-1]) if x_s is not None else None
        flat, flat_s = self.embed_in(flat, scalars=flat_s)
        C_mv = flat.shape[-2]
        h_mv = flat.reshape(B, N_obj, N_time, X, C_mv, 16)
        if flat_s is not None:
            C_s = flat_s.shape[-1]
            h_s = flat_s.reshape(B, N_obj, N_time, X, C_s)
        else:
            h_s = None

        # 4-block cycles
        for i in range(self.num_cycles):
            # GM over X
            h_mv, h_s = self.gm_blocks[2 * i](h_mv, h_s)
            # Object attention over N_obj
            h_mv, h_s = self.obj_attn_blocks[i](h_mv, h_s)
            # GM over X
            h_mv, h_s = self.gm_blocks[2 * i + 1](h_mv, h_s)
            # Time attention over N_time
            h_mv, h_s = self.time_attn_blocks[i](h_mv, h_s)

        # Output projection
        h_mv_flat = h_mv.reshape(-1, C_mv, 16)
        h_s_flat = h_s.reshape(-1, h_s.shape[-1]) if h_s is not None else None
        out_mv, out_s = self.embed_out(h_mv_flat, scalars=h_s_flat)
        out_mv = out_mv.reshape(*shape[:4], out_mv.shape[-2], 16)
        if out_s is not None:
            out_s = out_s.reshape(*shape[:4], out_s.shape[-1])

        return out_mv, out_s

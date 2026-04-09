"""Axial-split building blocks for Variant C.

The token sequence has shape [B, N_obj, N_time, X, C_mv, 16] and we alternate:
  1. GM mixing over the X (symmetry) axis
  2. Object-axis attention over N_obj
  3. GM mixing over X
  4. Time-axis attention over N_time

Tensor layout helpers move the active axis into the token position expected
by existing modules, then restore the original layout.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from gatr.primitives.normalization import equi_layer_norm
from finite_subgroup_gatr.layers.geo_mlp import GeoMLP
from finite_subgroup_gatr.layers.discrete_attention import DiscreteMVAttention
from finite_subgroup_gatr.pga.actions import construct_reference_multivector


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def move_axis_to_token_position(
    x_mv: Tensor,
    x_s: Tensor | None,
    axis: int,
) -> tuple[Tensor, Tensor | None, Any]:
    """Reshape so that axis `axis` (of the leading dims) is in position 1.

    Input shape:  [B, d0, d1, ..., C_mv, 16]
    Output shape: [B * (other dims), d_axis, C_mv, 16]

    Returns
    -------
    mv_flat : Tensor [B', X, C_mv, 16]
    s_flat  : Tensor [B', X, C_s] or None
    layout_info : dict with info to restore original layout
    """
    shape = x_mv.shape  # [B, d0, d1, ..., C_mv, 16]
    ndim_leading = x_mv.dim() - 2  # number of leading dims including B

    # Desired: bring axis (1-indexed among leading dims after B) to position 1
    # Axis 0 = B (batch), axis 1..ndim_leading-1 = spatial/structural dims, last two = C, 16

    # Permute to put axis at position 1 (right after B)
    # axis is 1-based among the structural dims d0, d1, ...
    perm = [0, axis + 1] + [i for i in range(1, ndim_leading) if i != axis + 1] + [ndim_leading, ndim_leading + 1]
    x_perm = x_mv.permute(*perm)

    B = shape[0]
    X = shape[axis + 1]
    C_mv = shape[-2]

    # Flatten all non-axis, non-MV dims into batch
    x_flat = x_perm.reshape(-1, X, C_mv, 16)

    if x_s is not None:
        # x_s: [B, d0, ..., C_s]
        perm_s = [0, axis + 1] + [i for i in range(1, ndim_leading) if i != axis + 1] + [ndim_leading]
        x_s_perm = x_s.permute(*perm_s[:-1] + [perm_s[-1] - 1] if x_s.dim() == x_mv.dim() - 1 else perm_s)
        # Simpler: just use the same leading permutation
        x_s_perm = x_s.permute(*perm[:-1])  # drop last (no component dim)
        s_flat = x_s_perm.reshape(-1, X, x_s.shape[-1])
    else:
        s_flat = None

    layout_info = {
        "orig_shape": shape,
        "orig_s_shape": x_s.shape if x_s is not None else None,
        "perm": perm,
        "axis": axis,
        "X": X,
        "B_flat": x_flat.shape[0],
    }
    return x_flat, s_flat, layout_info


def restore_axis_layout(
    x_mv: Tensor,
    x_s: Tensor | None,
    layout_info: Any,
) -> tuple[Tensor, Tensor | None]:
    """Inverse of move_axis_to_token_position."""
    orig_shape = layout_info["orig_shape"]
    perm = layout_info["perm"]
    orig_s_shape = layout_info["orig_s_shape"]
    C_mv = orig_shape[-2]

    # Restore the permuted shape
    permuted_shape = [orig_shape[p] for p in perm[:-2]] + [C_mv, 16]
    x_perm = x_mv.reshape(*permuted_shape)

    # Invert the permutation
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    x_restored = x_perm.permute(*inv_perm)

    if x_s is not None and orig_s_shape is not None:
        perm_s = perm[:-1]  # drop last dim
        inv_perm_s = [0] * len(perm_s)
        for i, p in enumerate(perm_s):
            inv_perm_s[p] = i
        s_shape = [orig_shape[p] for p in perm_s[:-1]] + [orig_s_shape[-1]]
        s_perm = x_s.reshape(*s_shape)
        s_restored = s_perm.permute(*inv_perm_s)
    else:
        s_restored = None

    return x_restored, s_restored


# ---------------------------------------------------------------------------
# Axial blocks
# ---------------------------------------------------------------------------

class SymmetryAxisGMBlock(nn.Module):
    """GM convolution along the symmetry (X) axis.

    Expected input layout: [B, ..., X, C_mv, 16]
    The symmetry axis is assumed to be the last structural axis (before C_mv and 16).

    Parameters
    ----------
    c_mv : int
    c_s : int or None
    backend : FiniteSymmetryBackend
    neighborhood_radius : int
    error_mode : str
    error_rank : int
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        backend,
        neighborhood_radius: int,
        error_mode: str = "none",
        error_rank: int = 4,
    ):
        super().__init__()
        from finite_subgroup_gatr.gm.token_mixer import GMTokenMixer
        self.gm_mixer = GMTokenMixer(
            backend, c_mv, c_s, neighborhood_radius, error_mode, error_rank,
        )
        self.geo_mlp = GeoMLP(c_mv, c_s)
        if c_s is not None:
            self.norm_s = nn.LayerNorm(c_s)
            self.norm_s2 = nn.LayerNorm(c_s)
        else:
            self.norm_s = self.norm_s2 = None

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Parameters
        ----------
        x_mv : Tensor [B, ..., X, C_mv, 16]
        x_s  : Tensor [B, ..., X, C_s] or None
        """
        # Flatten all leading dims except B and X into batch
        *lead, X, C_mv, _ = x_mv.shape
        B_lead = 1
        for d in lead:
            B_lead *= d
        mv_flat = x_mv.reshape(B_lead, X, C_mv, 16)
        s_flat = x_s.reshape(B_lead, X, x_s.shape[-1]) if x_s is not None else None

        # Norm + GM mixer
        h_mv = equi_layer_norm(mv_flat)
        h_s = self.norm_s(s_flat) if (s_flat is not None and self.norm_s is not None) else s_flat
        h_mv, h_s = self.gm_mixer(h_mv, h_s)
        mv_flat = mv_flat + h_mv
        if s_flat is not None and h_s is not None:
            s_flat = s_flat + h_s

        # Norm + GeoMLP
        h_mv = equi_layer_norm(mv_flat)
        h_s = self.norm_s2(s_flat) if (s_flat is not None and self.norm_s2 is not None) else s_flat
        ref_mv = construct_reference_multivector(h_mv)
        mlp_mv, mlp_s = self.geo_mlp(h_mv, h_s, ref_mv)
        mv_flat = mv_flat + mlp_mv
        if s_flat is not None and mlp_s is not None:
            s_flat = s_flat + mlp_s

        # Restore shape
        out_mv = mv_flat.reshape(*lead, X, C_mv, 16)
        out_s = s_flat.reshape(*lead, X, x_s.shape[-1]) if x_s is not None and s_flat is not None else x_s
        return out_mv, out_s


class _AxisAttentionBlock(nn.Module):
    """Attention over one named structural axis.

    Flattens all other leading dims into batch, applies attention over axis,
    then restores.
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        axis_position: int,
        num_heads: int = 4,
        use_scalar_logits: bool = True,
    ):
        super().__init__()
        self.axis_position = axis_position  # which leading dim (0=B, 1=first structural, ...)
        self.attention = DiscreteMVAttention(
            c_mv, c_s, num_heads=num_heads, use_scalar_logits=use_scalar_logits,
        )
        self.geo_mlp = GeoMLP(c_mv, c_s)
        if c_s is not None:
            self.norm_s = nn.LayerNorm(c_s)
            self.norm_s2 = nn.LayerNorm(c_s)
        else:
            self.norm_s = self.norm_s2 = None

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Parameters
        ----------
        x_mv : Tensor [B, d1, d2, ..., C_mv, 16]
            axis_position is a 0-indexed dimension (0=B, 1=d1, 2=d2, ...)
        """
        shape = x_mv.shape
        axis = self.axis_position  # target axis (absolute dim index, e.g. 1 or 2)
        C_mv = shape[-2]
        X = shape[axis]  # size of target axis
        B = shape[0]

        # Build permutation: bring axis to position 1
        # Structural dims are 1, 2, ..., ndim-3 (excluding B=0, C_mv=-2, 16=-1)
        struct_dims = list(range(1, x_mv.dim() - 2))  # e.g. [1, 2, 3] for 6D input
        other_struct = [d for d in struct_dims if d != axis]
        # perm: [0=B, axis, other_struct..., C_mv_dim, 16_dim]
        mv_dim = x_mv.dim() - 2  # index of C_mv dimension
        comp_dim = x_mv.dim() - 1  # index of 16 dimension
        perm = [0, axis] + other_struct + [mv_dim, comp_dim]
        x_perm = x_mv.permute(*perm)  # [B, X, rest, C_mv, 16]

        rest_size = 1
        for d in other_struct:
            rest_size *= shape[d]

        mv_flat = x_perm.reshape(B * rest_size, X, C_mv, 16)

        if x_s is not None:
            C_s = x_s.shape[-1]
            s_dim = x_s.dim() - 1
            perm_s = [0, axis] + other_struct + [s_dim]
            x_s_perm = x_s.permute(*perm_s)
            s_flat = x_s_perm.reshape(B * rest_size, X, C_s)
        else:
            s_flat = None

        # Norm + Attention
        h_mv = equi_layer_norm(mv_flat)
        h_s = self.norm_s(s_flat) if (s_flat is not None and self.norm_s is not None) else s_flat
        h_mv, h_s = self.attention(h_mv, h_s, mask=mask)
        mv_flat = mv_flat + h_mv
        if s_flat is not None and h_s is not None:
            s_flat = s_flat + h_s

        # Norm + GeoMLP
        h_mv = equi_layer_norm(mv_flat)
        h_s = self.norm_s2(s_flat) if (s_flat is not None and self.norm_s2 is not None) else s_flat
        ref_mv = construct_reference_multivector(h_mv)
        mlp_mv, mlp_s = self.geo_mlp(h_mv, h_s, ref_mv)
        mv_flat = mv_flat + mlp_mv
        if s_flat is not None and mlp_s is not None:
            s_flat = s_flat + mlp_s

        # Restore original shape: invert permutation
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i

        # perm brought shape to [B, X, rest, C_mv, 16]; after reshape it's [B*rest, X, C_mv, 16]
        # Reshape back to permuted shape then invert permutation
        permuted_lead = [B] + [shape[p] for p in perm[1:-2]] + [C_mv, 16]
        x_out = mv_flat.reshape(*permuted_lead).permute(*inv_perm)

        if x_s is not None and s_flat is not None:
            inv_perm_s = [0] * len(perm_s)
            for i, p in enumerate(perm_s):
                inv_perm_s[p] = i
            permuted_lead_s = [B] + [shape[p] for p in perm_s[1:-1]] + [x_s.shape[-1]]
            s_out = s_flat.reshape(*permuted_lead_s).permute(*inv_perm_s)
        else:
            s_out = x_s

        return x_out, s_out


class ObjectAxisMVAttentionBlock(_AxisAttentionBlock):
    """Attention over the object axis (leading dim 1)."""

    def __init__(self, c_mv: int, c_s: int | None, **kwargs):
        super().__init__(c_mv, c_s, axis_position=1, **kwargs)


class TimeAxisMVAttentionBlock(_AxisAttentionBlock):
    """Attention over the time axis (leading dim 2)."""

    def __init__(self, c_mv: int, c_s: int | None, **kwargs):
        super().__init__(c_mv, c_s, axis_position=2, **kwargs)

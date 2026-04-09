"""Output heads for decoding multivector outputs into geometric quantities."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gatr.layers.linear import EquiLinear
from gatr.interface.point import extract_point
from gatr.interface.rotation import extract_rotation


class ScalarHead(nn.Module):
    """Decode MV + scalar channels to scalar predictions.

    Parameters
    ----------
    c_mv : int
    c_s : int or None
    out_dim : int
    pool_tokens : bool
        If True, average over the token dimension before decoding.
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        out_dim: int,
        pool_tokens: bool = True,
    ):
        super().__init__()
        self.pool_tokens = pool_tokens
        in_s = (c_s or 0) + c_mv  # scalars + MV scalar components
        self.fc = nn.Linear(in_s, out_dim)

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x_mv : Tensor [B, X, C_mv, 16] or [B, ..., X, C_mv, 16]
        x_s  : Tensor [B, X, C_s] or None

        Returns
        -------
        Tensor [B, out_dim] (if pool_tokens) or [B, X, out_dim]
        """
        # Extract scalar components from MV: index 0 is the scalar grade
        mv_scalars = x_mv[..., 0]  # [..., X, C_mv]

        if x_s is not None:
            features = torch.cat([mv_scalars, x_s], dim=-1)  # [..., X, C_mv + C_s]
        else:
            features = mv_scalars

        if self.pool_tokens:
            features = features.mean(dim=-2)  # [..., C_mv + C_s]

        return self.fc(features)


class PointHead(nn.Module):
    """Decode MV outputs to 3D point coordinates.

    Extracts points from the trivector (grade-3) components.

    Parameters
    ----------
    c_mv : int
    out_points : int
        Number of points to predict.
    pool_tokens : bool
    """

    def __init__(
        self,
        c_mv: int,
        out_points: int = 1,
        pool_tokens: bool = True,
    ):
        super().__init__()
        self.pool_tokens = pool_tokens
        self.linear = EquiLinear(c_mv, out_points)

    def forward(self, x_mv: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x_mv : Tensor [B, X, C_mv, 16]

        Returns
        -------
        Tensor [B, out_points, 3] or [B, X, out_points, 3]
        """
        if self.pool_tokens:
            x_mv = x_mv.mean(dim=-3)  # [B, C_mv, 16]

        out_mv, _ = self.linear(x_mv)  # [..., out_points, 16]
        coords = extract_point(out_mv)  # [..., out_points, 3]
        return coords


class MultivectorHead(nn.Module):
    """Project hidden MV channels to a target number of output MV channels.

    Parameters
    ----------
    c_mv : int
    c_s : int or None
    out_mv_channels : int
    out_s_channels : int or None
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        out_mv_channels: int,
        out_s_channels: int | None = None,
    ):
        super().__init__()
        self.linear = EquiLinear(
            c_mv, out_mv_channels,
            in_s_channels=c_s,
            out_s_channels=out_s_channels,
        )

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        return self.linear(x_mv, scalars=x_s)

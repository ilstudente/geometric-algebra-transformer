"""Shared GeoMLP for all three variants.

GeoBilinear(x, y; z) = Concat(x*y, EquiJoin(x, y; z))

followed by gated nonlinearity and output projection.
Grade-aware channel mixing is enforced when grade_aware=True.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from gatr.layers.linear import EquiLinear
from gatr.primitives.bilinear import geometric_product
from gatr.primitives.dual import equivariant_join
from gatr.primitives.nonlinearities import gated_gelu
from gatr.primitives.normalization import equi_layer_norm


@dataclass
class GeoMLPConfig:
    c_mv: int = 8
    c_s: int = 32
    hidden_mv: int = 16
    hidden_s: int = 64
    grade_aware: bool = True


class GeoMLP(nn.Module):
    """Geometric MLP combining geometric product and equivariant join.

    Architecture (following GATr's GeometricBilinear + nonlinearity + projection):

    1. Linear projections for GP: left, right -> [hidden_mv/2]
    2. Linear projections for join: left, right -> [hidden_mv/2]
    3. GP branch: gp(left_gp, right_gp)
    4. Join branch: join(left_join, right_join, ref)
    5. Concatenate: [hidden_mv]
    6. Gated GELU nonlinearity
    7. Output EquiLinear projection: [hidden_mv] -> [c_mv]
    8. Scalar path: MLP(c_s -> hidden_s -> c_s)

    Parameters
    ----------
    c_mv : int
    c_s : int or None
    hidden_mv : int
        Should be even (split equally between GP and join branches).
    hidden_s : int or None
    grade_aware : bool
        Ignored for now (grade awareness is implicit in EquiLinear).
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        hidden_mv: int | None = None,
        hidden_s: int | None = None,
        grade_aware: bool = True,
        scalar_ffn: "nn.Module | None" = None,
    ):
        super().__init__()
        if hidden_mv is None:
            hidden_mv = 2 * c_mv
        if hidden_s is None:
            hidden_s = 2 * c_s if c_s is not None else None

        assert hidden_mv % 2 == 0, "hidden_mv must be even"
        half = hidden_mv // 2

        # Scalar layer norm (MV norm is applied via equi_layer_norm function)
        if c_s is not None:
            self.s_norm = nn.LayerNorm(c_s)
        else:
            self.s_norm = None

        # GP branch projections
        self.linear_gp_left = EquiLinear(
            c_mv, half, in_s_channels=c_s, out_s_channels=None,
        )
        _right_init = "almost_unit_scalar" if c_s is not None else "default"
        self.linear_gp_right = EquiLinear(
            c_mv, half, in_s_channels=c_s, out_s_channels=None,
            initialization=_right_init,
        )

        # Join branch projections
        self.linear_join_left = EquiLinear(
            c_mv, half, in_s_channels=c_s, out_s_channels=None,
        )
        self.linear_join_right = EquiLinear(
            c_mv, half, in_s_channels=c_s, out_s_channels=None,
        )

        # Gating: scalars that gate the bilinear output
        # We use EquiLinear to produce the gate from MV + scalars
        self.gate_linear = EquiLinear(
            hidden_mv, hidden_mv, in_s_channels=None, out_s_channels=None,
        )

        # Output projection
        self.linear_out = EquiLinear(
            hidden_mv, c_mv, in_s_channels=c_s, out_s_channels=c_s,
        )

        # Scalar MLP — use injected bundle if provided, else build dense default
        if scalar_ffn is not None:
            self.scalar_mlp = scalar_ffn
        elif c_s is not None and hidden_s is not None:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(c_s, hidden_s),
                nn.GELU(),
                nn.Linear(hidden_s, c_s),
            )
        else:
            self.scalar_mlp = None

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None,
        ref_mv: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_mv : Tensor [..., C_mv, 16]
        x_s  : Tensor [..., C_s] or None
        ref_mv : Tensor [..., 1, 16]  reference multivector for join

        Returns
        -------
        out_mv : Tensor [..., C_mv, 16]
        out_s  : Tensor [..., C_s] or None
        """
        # Layer norm
        h_mv = equi_layer_norm(x_mv)
        h_s = self.s_norm(x_s) if (x_s is not None and self.s_norm is not None) else x_s

        # GP branch
        gp_left, _ = self.linear_gp_left(h_mv, scalars=h_s)
        gp_right, _ = self.linear_gp_right(h_mv, scalars=h_s)
        gp_out = geometric_product(gp_left, gp_right)  # [..., half, 16]

        # Join branch
        j_left, _ = self.linear_join_left(h_mv, scalars=h_s)
        j_right, _ = self.linear_join_right(h_mv, scalars=h_s)
        join_out = equivariant_join(j_left, j_right, ref_mv)  # [..., half, 16]

        # Concatenate along channel dim
        bilinear_out = torch.cat([gp_out, join_out], dim=-2)  # [..., hidden_mv, 16]

        # Gated GELU: use scalar component as gates
        gates, _ = self.gate_linear(bilinear_out)
        gate_values = gates[..., [0]]  # [..., hidden_mv, 1]  scalar component
        gated = gated_gelu(bilinear_out, gate_values)  # [..., hidden_mv, 16]

        # Output projection
        out_mv, out_s = self.linear_out(gated, scalars=h_s)

        # Scalar residual path
        if x_s is not None and self.scalar_mlp is not None:
            if out_s is not None:
                out_s = out_s + self.scalar_mlp(x_s)
            else:
                out_s = self.scalar_mlp(x_s)

        return out_mv, out_s

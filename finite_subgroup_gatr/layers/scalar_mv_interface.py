"""ScalarMVScalarInterface: GM-structured scalar pre-projection for mixed layers.

GATr's EquiLinear layers accept both multivector and auxiliary scalar channels,
mixing them internally.  This module wraps the scalar side of that interface:
it applies a GM-structured (or dense) pre-projection to auxiliary scalars before
they are passed into EquiLinear, and an optional GM-structured post-projection
to the scalar output.

Design principle
----------------
The multivector algebraic operators (EquiLinear, geometric product, join) are
**not** modified.  Only the scalar half of the scalar↔MV mixing is touched.
This preserves GATr's PGA geometry while injecting GM structure into the
non-geometric side-channel.

Usage
-----
In a GeoMLP-like layer::

    # Before: direct call
    out_mv, out_s = equi_linear(h_mv, scalars=h_s)

    # After: pre-process scalars with this interface
    h_s_pre = interface.forward_pre(h_s)
    out_mv, out_s_raw = equi_linear(h_mv, scalars=h_s_pre)
    out_s = interface.forward_post(out_s_raw)  # optional post-projection

"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.layers.scalar_gm_linear import ScalarGMLinear


class ScalarMVScalarInterface(nn.Module):
    """GM-structured scalar pre/post projections for EquiLinear interfaces.

    Parameters
    ----------
    backend : FiniteSymmetryBackend or None
        Required when mode != "standard".
    c_s_in : int
        Scalar channels entering the interface (pre-projection input dim).
    c_s_out : int
        Scalar channels leaving the interface (post-projection output dim).
        Set equal to c_s_in if the EquiLinear does not change scalar width.
    neighborhood_radius : int
    mode : {"standard", "gm_exact", "gm_approx"}
        "standard"  – no-op; interface is a pure pass-through.
        "gm_exact"  – ScalarGMLinear pre-projection with error_mode="none".
        "gm_approx" – ScalarGMLinear pre-projection with given error_mode.
    error_mode : {"none", "low_rank", "dense_penalized"}
    error_rank : int
    apply_post : bool
        If True, also apply a GM-structured post-projection to the scalar output
        of EquiLinear (after the MV-scalar mixing has happened).
    """

    def __init__(
        self,
        backend: FiniteSymmetryBackend | None,
        c_s_in: int,
        c_s_out: int,
        neighborhood_radius: int = 1,
        mode: str = "standard",
        error_mode: str = "none",
        error_rank: int = 4,
        apply_post: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.apply_post = apply_post

        if mode != "standard" and backend is None:
            raise ValueError(
                f"ScalarMVScalarInterface: backend required for mode='{mode}'"
            )

        actual_error_mode = "none" if mode == "gm_exact" else error_mode

        if mode == "standard":
            self.pre_proj: nn.Module | None = None
            self.post_proj: nn.Module | None = None
        else:
            self.pre_proj = ScalarGMLinear(
                backend,        # type: ignore[arg-type]
                c_in=c_s_in,
                c_out=c_s_in,   # same width — projection within scalar space
                neighborhood_radius=neighborhood_radius,
                error_mode=actual_error_mode,
                error_rank=error_rank,
            )
            if apply_post:
                self.post_proj = ScalarGMLinear(
                    backend,    # type: ignore[arg-type]
                    c_in=c_s_out,
                    c_out=c_s_out,
                    neighborhood_radius=neighborhood_radius,
                    error_mode=actual_error_mode,
                    error_rank=error_rank,
                )
            else:
                self.post_proj = None

    def forward_pre(self, x_s: Tensor | None) -> Tensor | None:
        """Apply GM pre-projection to scalars before EquiLinear.

        Parameters
        ----------
        x_s : Tensor [..., X, C_s] or None

        Returns
        -------
        Tensor [..., X, C_s] or None
        """
        if x_s is None or self.pre_proj is None:
            return x_s
        return self.pre_proj(x_s)

    def forward_post(self, x_s: Tensor | None) -> Tensor | None:
        """Apply GM post-projection to scalar output of EquiLinear (optional).

        Parameters
        ----------
        x_s : Tensor [..., X, C_s] or None

        Returns
        -------
        Tensor [..., X, C_s] or None
        """
        if x_s is None or self.post_proj is None:
            return x_s
        return self.post_proj(x_s)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def error_norm(self) -> Tensor:
        """Total residual error Frobenius norm."""
        total = torch.zeros(1)
        for m in [self.pre_proj, self.post_proj]:
            if isinstance(m, ScalarGMLinear):
                total = total + m.error_norm().cpu()
        return total

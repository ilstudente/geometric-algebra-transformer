"""ScalarAttentionProjectionBundle: GM-structured scalar Q/K/V projections.

Replaces the dense ``nn.Linear`` Q/K (and optionally V/output) projections
in DiscreteMVAttention with ScalarGMLinear layers.

The bundle is constructed once and injected into DiscreteMVAttention via its
``scalar_attn_bundle`` constructor parameter.

Modes
-----
"dense"     Standard dense linear layers (default).
"gm_exact"  ScalarGMLinear with error_mode="none".
"gm_approx" ScalarGMLinear with the given error_mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.layers.scalar_gm_linear import ScalarGMLinear


class ScalarAttentionProjectionBundle(nn.Module):
    """Scalar Q/K/V projection bundle with optional GM structure.

    Parameters
    ----------
    backend : FiniteSymmetryBackend or None
        Required when mode != "dense".
    c_s : int
        Input scalar channels.
    num_heads : int
    head_dim_s : int
        Scalar dimensions per head.  Total Q/K output = num_heads * head_dim_s.
    mode : {"dense", "gm_exact", "gm_approx"}
    neighborhood_radius : int
    error_mode : {"none", "low_rank", "dense_penalized"}
    error_rank : int
    project_values : bool
        If True, also build a scalar value projection (c_s → c_s).
    project_output : bool
        If True, also build a scalar output projection (c_s → c_s).
    """

    def __init__(
        self,
        backend: FiniteSymmetryBackend | None,
        c_s: int,
        num_heads: int,
        head_dim_s: int,
        mode: str = "dense",
        neighborhood_radius: int = 1,
        error_mode: str = "none",
        error_rank: int = 4,
        project_values: bool = False,
        project_output: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim_s = head_dim_s
        self.project_values = project_values
        self.project_output = project_output

        total_qk_s = num_heads * head_dim_s

        if mode != "dense" and backend is None:
            raise ValueError(
                f"ScalarAttentionProjectionBundle: backend required for mode='{mode}'"
            )

        actual_error_mode = "none" if mode == "gm_exact" else error_mode

        def _make(c_in: int, c_out: int) -> nn.Module:
            if mode == "dense":
                return nn.Linear(c_in, c_out)
            return ScalarGMLinear(
                backend,        # type: ignore[arg-type]
                c_in=c_in,
                c_out=c_out,
                neighborhood_radius=neighborhood_radius,
                error_mode=actual_error_mode,
                error_rank=error_rank,
            )

        self.q_proj: nn.Module = _make(c_s, total_qk_s)
        self.k_proj: nn.Module = _make(c_s, total_qk_s)

        if project_values:
            self.v_proj: nn.Module | None = _make(c_s, c_s)
        else:
            self.v_proj = None

        if project_output:
            self.out_proj: nn.Module | None = _make(c_s, c_s)
        else:
            self.out_proj = None

    # ------------------------------------------------------------------
    # Forward helpers called by DiscreteMVAttention
    # ------------------------------------------------------------------

    def forward_q(self, s: Tensor) -> Tensor:
        """Project scalars to Q.  [..., X, C_s] → [..., X, H*head_dim_s]"""
        return self.q_proj(s)

    def forward_k(self, s: Tensor) -> Tensor:
        """Project scalars to K.  [..., X, C_s] → [..., X, H*head_dim_s]"""
        return self.k_proj(s)

    def forward_v(self, s: Tensor) -> Tensor | None:
        """Project scalars to V.  Only active when project_values=True."""
        if self.v_proj is None:
            return None
        return self.v_proj(s)

    def forward_out(self, s: Tensor) -> Tensor | None:
        """Project scalar attention output.  Only active when project_output=True."""
        if self.out_proj is None:
            return None
        return self.out_proj(s)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def error_norm(self) -> Tensor:
        """Total residual error Frobenius norm."""
        total = torch.zeros(1)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if isinstance(m, ScalarGMLinear):
                total = total + m.error_norm().cpu()
        return total

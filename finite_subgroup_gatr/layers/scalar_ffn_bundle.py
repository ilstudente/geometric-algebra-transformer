"""ScalarFFNBundle: configurable scalar feedforward network.

Replaces the dense ``nn.Sequential(Linear, GELU, Linear)`` inside GeoMLP with a
GM-structured or dense scalar MLP.  All three modes share the same call
signature so callers do not need to know which mode is active.

Modes
-----
"dense"     Standard dense linear layers (default; backward-compatible).
"gm_exact"  ScalarGMLinear with error_mode="none" (exact equivariance).
"gm_approx" ScalarGMLinear with the given error_mode (low_rank or dense_penalized).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.layers.scalar_gm_linear import ScalarGMLinear


class ScalarFFNBundle(nn.Module):
    """Scalar feedforward network with optional GM structure.

    Parameters
    ----------
    backend : FiniteSymmetryBackend or None
        Required when mode != "dense".
    c_s : int
        Input and output scalar channels.
    hidden_s : int
        Hidden scalar channels.
    mode : {"dense", "gm_exact", "gm_approx"}
    neighborhood_radius : int
    error_mode : {"none", "low_rank", "dense_penalized"}
        Used when mode == "gm_approx".
    error_rank : int
    activation : {"gelu", "relu"}
    dropout : float
    """

    def __init__(
        self,
        backend: FiniteSymmetryBackend | None,
        c_s: int,
        hidden_s: int,
        mode: str = "dense",
        neighborhood_radius: int = 1,
        error_mode: str = "none",
        error_rank: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.c_s = c_s
        self.hidden_s = hidden_s

        if mode != "dense" and backend is None:
            raise ValueError(
                f"ScalarFFNBundle: backend required for mode='{mode}'"
            )

        # Activation
        act: nn.Module
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        # Optional dropout
        drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if mode == "dense":
            self.fc1: nn.Module = nn.Linear(c_s, hidden_s)
            self.fc2: nn.Module = nn.Linear(hidden_s, c_s)
        elif mode in ("gm_exact", "gm_approx"):
            actual_error_mode = "none" if mode == "gm_exact" else error_mode
            self.fc1 = ScalarGMLinear(
                backend,        # type: ignore[arg-type]
                c_in=c_s,
                c_out=hidden_s,
                neighborhood_radius=neighborhood_radius,
                error_mode=actual_error_mode,
                error_rank=error_rank,
            )
            self.fc2 = ScalarGMLinear(
                backend,        # type: ignore[arg-type]
                c_in=hidden_s,
                c_out=c_s,
                neighborhood_radius=neighborhood_radius,
                error_mode=actual_error_mode,
                error_rank=error_rank,
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'dense', 'gm_exact', or 'gm_approx'.")

        self.act = act
        self.drop = drop

    def forward(self, x: Tensor) -> Tensor:
        """Apply feedforward transform.

        Parameters
        ----------
        x : Tensor [..., X, C_s] for GM modes, or [..., C_s] for dense mode.
            Dense mode expects the token axis to be folded into the batch dims.

        Returns
        -------
        Tensor same shape as input.
        """
        return self.fc2(self.drop(self.act(self.fc1(x))))

    # ------------------------------------------------------------------
    # Diagnostic helpers (pass-through to ScalarGMLinear when applicable)
    # ------------------------------------------------------------------

    def error_norm(self) -> Tensor:
        """Total residual error Frobenius norm (zero for dense mode)."""
        if self.mode == "dense":
            return torch.zeros(1)
        total = torch.zeros(1)
        for m in [self.fc1, self.fc2]:
            if isinstance(m, ScalarGMLinear):
                total = total + m.error_norm().cpu()
        return total

    def displacement_proxy(self) -> Tensor:
        """Structural deviation proxy (zero for dense mode)."""
        if self.mode == "dense":
            return torch.zeros(1)
        total = torch.zeros(1)
        for m in [self.fc1, self.fc2]:
            if isinstance(m, ScalarGMLinear):
                total = total + m.displacement_proxy().cpu()
        return total

"""ScalarGMLinear: GM-structured linear map for scalar token sequences.

Implements the group convolution:

    s'(x) = sum_{g in N_k} W_g * s(x * g)   +   bias

where:
  - x ranges over the finite token domain X = G_f (or any axis indexed by it)
  - W_g are per-neighborhood-element channel-mixing matrices (with optional residuals)
  - x * g is a right-shift, implemented as an index gather over the token axis (dim=-2)

The input/output shape is [..., X, C], preserving all leading axes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.gm.approximation import GMMixingWeights
from finite_subgroup_gatr.gm.shifts import precompute_shift_permutations


class ScalarGMLinear(nn.Module):
    """GM-structured linear map for scalar token sequences.

    Operates on ``[..., X, C_in]`` → ``[..., X, C_out]`` where ``X = |G_f|``.
    All leading axes (batch, object, time, …) are preserved by broadcasting.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    c_in : int
    c_out : int
    neighborhood_radius : int
        Word-metric radius; determines the neighborhood N_k and hence K = |N_k|.
    bias : bool
        Single shared bias added after accumulation (not per-neighborhood-element).
    error_mode : {"none", "low_rank", "dense_penalized"}
        "none"            – exact GM weights, no residual.
        "low_rank"        – W_g = W_g_GM + U_g V_g^T, rank = error_rank.
        "dense_penalized" – W_g = W_g_GM + E_g, full dense residual.
    error_rank : int
        Rank for low_rank mode.
    share_across_channels : bool
        If True a single weight matrix is shared for ALL neighborhood elements.
        This breaks equivariance but gives a maximally parameter-efficient baseline.
    """

    def __init__(
        self,
        backend: FiniteSymmetryBackend,
        c_in: int,
        c_out: int,
        neighborhood_radius: int,
        bias: bool = True,
        error_mode: str = "none",
        error_rank: int = 4,
        share_across_channels: bool = False,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.error_mode = error_mode
        self.share_across_channels = share_across_channels

        # Neighborhood indices [K] and shift permutations [K, X]
        nbr_indices = backend.neighborhood(neighborhood_radius)
        self.register_buffer("nbr_indices", nbr_indices)
        self.K = len(nbr_indices)

        perms = precompute_shift_permutations(backend, nbr_indices)
        self.register_buffer("perms", perms)  # [K, X]

        # Weight bank – one ApproxLinear per neighbourhood element
        if share_across_channels:
            # Single shared matrix; store as a plain parameter to avoid bias
            self.shared_weight = nn.Parameter(torch.empty(c_out, c_in))
            nn.init.kaiming_uniform_(self.shared_weight, a=5 ** 0.5)
            self.gm_weights = None
        else:
            self.shared_weight = None
            self.gm_weights = GMMixingWeights(self.K, c_in, c_out, error_mode, error_rank)

        # Single shared bias (not per-element)
        if bias:
            self.bias = nn.Parameter(torch.zeros(c_out))
        else:
            self.register_parameter("bias", None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Apply GM linear map.

        Parameters
        ----------
        x : Tensor [..., X, C_in]

        Returns
        -------
        Tensor [..., X, C_out]
        """
        out = x.new_zeros(*x.shape[:-1], self.c_out)

        for k in range(self.K):
            perm = self.perms[k]                    # [X]
            x_shifted = x[..., perm, :]             # [..., X, C_in]

            if self.share_across_channels:
                W = self.shared_weight              # [C_out, C_in]
            else:
                W = self.gm_weights.weights[k].effective_weight()  # [C_out, C_in]

            out = out + x_shifted @ W.T             # [..., X, C_out]

        if self.bias is not None:
            out = out + self.bias

        return out

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def error_norm(self) -> Tensor:
        """Total Frobenius norm of all residual error terms."""
        if self.gm_weights is not None:
            return self.gm_weights.error_norm()
        return torch.zeros(1, device=self._device())

    def displacement_proxy(self) -> Tensor:
        """Proxy for structural deviation from exact equivariance."""
        if self.gm_weights is not None:
            return self.gm_weights.displacement_proxy()
        return torch.zeros(1, device=self._device())

    def _device(self):
        p = next(self.parameters(), None)
        b = next(self.buffers(), None)
        if p is not None:
            return p.device
        if b is not None:
            return b.device
        return torch.device("cpu")

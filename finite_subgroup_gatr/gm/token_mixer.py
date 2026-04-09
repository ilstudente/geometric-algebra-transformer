"""GM token mixer: group-equivariant convolution over the finite token domain.

GMConv(z)(x) = sum_{g in N_k} W_g * z(g^{-1} * x)

where:
  - N_k is the word-metric ball of radius k
  - W_g is a [C_out, C_in] channel-mixing matrix (possibly with approximate residual)
  - g^{-1} * x is implemented as an index gather over the token dimension

W_g mixes channels only; it does not mix the 16 multivector components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.gm.approximation import GMMixingWeights
from finite_subgroup_gatr.gm.shifts import precompute_shift_permutations


class GMTokenMixer(nn.Module):
    """Group-equivariant token mixer for finite groups.

    Operates on sequences of length X = |G_f| by performing a group convolution
    over the word-metric neighborhood of the identity.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    c_mv : int
        Multivector channels.
    c_s : int or None
        Scalar channels.
    neighborhood_radius : int
    error_mode : {"none", "low_rank", "dense_penalized"}
    error_rank : int
    """

    def __init__(
        self,
        backend: FiniteSymmetryBackend,
        c_mv: int,
        c_s: int | None,
        neighborhood_radius: int,
        error_mode: str = "none",
        error_rank: int = 4,
    ):
        super().__init__()
        self.backend = backend
        self.c_mv = c_mv
        self.c_s = c_s
        self.error_mode = error_mode

        # Neighborhood
        nbr_indices = backend.neighborhood(neighborhood_radius)
        self.register_buffer("nbr_indices", nbr_indices)
        K = len(nbr_indices)
        self.K = K

        # Precompute permutation tables [K, X]
        X = backend.size()
        perms = precompute_shift_permutations(backend, nbr_indices)
        self.register_buffer("perms", perms)  # [K, X]

        # Channel mixing weights
        # MV channels: one weight matrix per neighborhood element
        self.mv_weights = GMMixingWeights(K, c_mv, c_mv, error_mode, error_rank)

        # Scalar channel mixing (if applicable)
        if c_s is not None:
            self.s_weights = GMMixingWeights(K, c_s, c_s, error_mode, error_rank)
        else:
            self.s_weights = None

        # Scalar output layer norm
        if c_s is not None:
            self.s_norm = nn.LayerNorm(c_s)
        else:
            self.s_norm = None

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_mv : Tensor [B, X, C_mv, 16]
        x_s  : Tensor [B, X, C_s] or None

        Returns
        -------
        out_mv : Tensor [B, X, C_mv, 16]
        out_s  : Tensor [B, X, C_s] or None
        """
        B, X, C_mv, _ = x_mv.shape
        K = self.K

        # Accumulate outputs
        out_mv = torch.zeros_like(x_mv)
        out_s = torch.zeros_like(x_s) if x_s is not None else None

        # Apply the K group shifts and mix channels
        for k in range(K):
            perm = self.perms[k]  # [X]

            # Shift: gather along the token dimension
            x_shifted_mv = x_mv[:, perm, :, :]  # [B, X, C_mv, 16]

            # Mix channels: W_k @ x_shifted  (over channel dimension only)
            # x_shifted_mv: [B, X, C_mv, 16]
            # effective_weight: [C_mv, C_mv]
            W = self.mv_weights.weights[k].effective_weight()  # [C_mv, C_mv]
            # Einsum: y_oc = sum_{ic} W[oc, ic] * x[ic]
            mixed_mv = torch.einsum("oi, bxid -> bxod", W, x_shifted_mv)
            out_mv = out_mv + mixed_mv

            if x_s is not None and self.s_weights is not None:
                x_shifted_s = x_s[:, perm, :]  # [B, X, C_s]
                W_s = self.s_weights.weights[k].effective_weight()  # [C_s, C_s]
                mixed_s = torch.einsum("oi, bxi -> bxo", W_s, x_shifted_s)
                out_s = out_s + mixed_s

        return out_mv, out_s

    def error_norm(self) -> Tensor:
        """Total Frobenius norm of residual error terms."""
        total = self.mv_weights.error_norm()
        if self.s_weights is not None:
            total = total + self.s_weights.error_norm()
        return total

    def displacement_proxy(self) -> Tensor:
        """Proxy for structural deviation from exact equivariance."""
        return self.mv_weights.displacement_proxy()

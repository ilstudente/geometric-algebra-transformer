"""Subgroup pooling for GM architectures.

Pools a token sequence over the cosets of a subgroup H <= G_f,
producing a coarser token sequence indexed by G_f / H.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend


class SubgroupPool(nn.Module):
    """Pool (average) tokens over left cosets of a subgroup.

    Given a token sequence x: [B, X, C_mv, 16] with X = |G_f|,
    averages over the coset H to produce x_pooled: [B, |G_f/H|, C_mv, 16].

    Parameters
    ----------
    backend : FiniteSymmetryBackend  (the full group G_f)
    subgroup : FiniteSymmetryBackend (the subgroup H)
    """

    def __init__(
        self,
        backend: FiniteSymmetryBackend,
        subgroup: FiniteSymmetryBackend,
    ):
        super().__init__()
        self.backend = backend
        self.subgroup = subgroup

        cosets = backend.coset_partition(subgroup)
        self.num_cosets = len(cosets)

        # Build coset index matrix: [num_cosets, coset_size]
        coset_size = len(cosets[0])
        coset_indices = torch.zeros(self.num_cosets, coset_size, dtype=torch.long)
        for i, coset in enumerate(cosets):
            if len(coset) != coset_size:
                raise ValueError("Cosets have different sizes; subgroup is not normal")
            coset_indices[i] = torch.tensor(coset)
        self.register_buffer("coset_indices", coset_indices)

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Average-pool tokens over cosets.

        Parameters
        ----------
        x_mv : Tensor [B, X, C_mv, 16]
        x_s  : Tensor [B, X, C_s] or None

        Returns
        -------
        out_mv : Tensor [B, num_cosets, C_mv, 16]
        out_s  : Tensor [B, num_cosets, C_s] or None
        """
        B = x_mv.shape[0]
        N_c = self.num_cosets
        coset_size = self.coset_indices.shape[1]
        C_mv = x_mv.shape[-2]

        # Gather all coset elements at once using flat indexing
        flat_idx = self.coset_indices.reshape(-1)  # [N_c * coset_size]
        x_gathered = x_mv[:, flat_idx, :, :]  # [B, N_c*coset_size, C_mv, 16]
        out_mv = x_gathered.reshape(B, N_c, coset_size, C_mv, 16).mean(dim=2)

        if x_s is not None:
            C_s = x_s.shape[-1]
            s_gathered = x_s[:, flat_idx, :]  # [B, N_c*coset_size, C_s]
            out_s = s_gathered.reshape(B, N_c, coset_size, C_s).mean(dim=2)
        else:
            out_s = None

        return out_mv, out_s

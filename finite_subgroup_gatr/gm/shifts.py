"""Group-shift (gather) operations for GM convolution.

A group shift by g maps token x to token g^{-1}*x (or g*x for right shifts).
These are implemented as index-gather operations over the token dimension.
"""

from __future__ import annotations

import torch
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend


def group_shift(
    x_mv: Tensor,
    perm: Tensor,
    x_s: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Apply a token permutation (group shift) to an MVTensor.

    Parameters
    ----------
    x_mv : Tensor [B, X, C_mv, 16]
    perm : LongTensor [X]
        perm[i] = new source index for position i.
    x_s : Tensor [B, X, C_s] or None

    Returns
    -------
    Tensor [B, X, C_mv, 16], Tensor [B, X, C_s] or None
    """
    mv_out = x_mv[:, perm, :, :]
    s_out = x_s[:, perm, :] if x_s is not None else None
    return mv_out, s_out


def group_shift_batch(
    x_mv: Tensor,
    perms: Tensor,
    x_s: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Apply a batch of token permutations and stack the results.

    Parameters
    ----------
    x_mv : Tensor [B, X, C_mv, 16]
    perms : LongTensor [K, X]
        K permutations to apply simultaneously.
    x_s : Tensor [B, X, C_s] or None

    Returns
    -------
    Tensor [B, K, X, C_mv, 16], Tensor [B, K, X, C_s] or None
    """
    # perms: [K, X]
    # x_mv:  [B, X, C_mv, 16]
    K, X = perms.shape
    B, _, C_mv, _ = x_mv.shape

    # Expand perm to gather over token dim
    # x_mv[:, perms[k], :, :] for each k
    # Use advanced indexing
    idx = perms.unsqueeze(0).expand(B, K, X)  # [B, K, X]
    # Expand x_mv to [B, 1, X, C_mv, 16] then gather
    x_exp = x_mv.unsqueeze(1).expand(B, K, X, C_mv, 16)
    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, K, X, C_mv, 16)
    mv_out = torch.gather(x_exp, 2, idx_exp)  # [B, K, X, C_mv, 16]

    if x_s is not None:
        C_s = x_s.shape[-1]
        x_s_exp = x_s.unsqueeze(1).expand(B, K, X, C_s)
        idx_s = idx.unsqueeze(-1).expand(B, K, X, C_s)
        s_out = torch.gather(x_s_exp, 2, idx_s)  # [B, K, X, C_s]
    else:
        s_out = None

    return mv_out, s_out


def precompute_shift_permutations(
    backend: FiniteSymmetryBackend,
    neighborhood_indices: Tensor,
) -> Tensor:
    """Precompute RIGHT-shift permutation tables for all elements in the neighborhood.

    For left-equivariance under group action ρ(h)f(x) = f(h^{-1}*x), the correct
    equivariant group convolution uses RIGHT shifts:
        out(x) = sum_{g in N_k} W_g * in(x * g)

    For each g in the neighborhood, the permutation maps x -> mul_table[x, g] = x*g.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    neighborhood_indices : LongTensor [K]

    Returns
    -------
    LongTensor [K, X]  where X = backend.size()
        perms[k, x] = index of element(x) * g_k
    """
    X = backend.size()
    K = len(neighborhood_indices)
    perms = torch.zeros(K, X, dtype=torch.long)

    for k, g_idx in enumerate(neighborhood_indices.tolist()):
        # Right shift by g: perm[x] = x * g = mul_table[x, g]
        perms[k] = backend.mul_table[:, g_idx]

    return perms

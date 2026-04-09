"""PGA group-action helpers and reference-multivector utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend


def apply_group_action_to_multivectors(
    backend: FiniteSymmetryBackend,
    g: int,
    mv: Tensor,
) -> Tensor:
    """Apply finite group element g to a batch of PGA multivectors.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    g : int
    mv : Tensor [..., 16]

    Returns
    -------
    Tensor [..., 16]
    """
    return backend.action_on_multivectors(g, mv)


def apply_group_action_to_tokens(
    backend: FiniteSymmetryBackend,
    g: int,
    x_mv: Tensor,
    x_s: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Permute the token axis of an MVTensor by group element g.

    The token axis is assumed to be the second-to-last axis before the channel
    dimension, i.e.:
        x_mv : [B, X, C_mv, 16]
        x_s  : [B, X, C_s]

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    g : int
    x_mv : Tensor [B, X, C_mv, 16]
    x_s  : Tensor [B, X, C_s] or None

    Returns
    -------
    Tensor [B, X, C_mv, 16], Tensor [B, X, C_s] or None
    """
    X = x_mv.shape[-3]
    token_indices = torch.arange(X, device=x_mv.device)
    permuted = backend.action_on_group_tokens(g, token_indices)

    mv_out = x_mv[:, permuted, :, :]
    s_out = x_s[:, permuted, :] if x_s is not None else None
    return mv_out, s_out


def construct_reference_multivector(mv: Tensor) -> Tensor:
    """Construct a reference multivector from the mean of input multivectors.

    The reference is used to break the orientation ambiguity in the equivariant
    join.  Following GATr, we take the mean over channels (and optionally tokens).

    Parameters
    ----------
    mv : Tensor [..., C, 16]
        Input multivectors.

    Returns
    -------
    ref : Tensor [..., 1, 16]
        Reference multivector (broadcastable over channels).
    """
    ref = mv.mean(dim=-2, keepdim=True)
    return ref


def equivariance_error(
    backend: FiniteSymmetryBackend,
    model_fn,
    x_mv: Tensor,
    x_s: Tensor | None,
    num_samples: int = 8,
) -> Tensor:
    """Measure the finite-group equivariance error of a model.

    Computes:
        E[g] = || f(T_g x) - T_g f(x) ||^2  averaged over g

    where T_g acts on both token domain and multivector components.

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    model_fn : callable (x_mv, x_s) -> (y_mv, y_s)
    x_mv : Tensor [B, X, C_mv, 16]
    x_s  : Tensor [B, X, C_s] or None
    num_samples : int
        Number of group elements to sample.

    Returns
    -------
    Tensor scalar
    """
    G = backend.size()
    indices = torch.randperm(G)[:num_samples]

    total_err = 0.0
    for g_idx in indices.tolist():
        # Apply g to input (token permutation)
        x_mv_g, x_s_g = apply_group_action_to_tokens(backend, g_idx, x_mv, x_s)
        # Apply g to each multivector in the token-permuted input
        M_g = backend.mv_rep[g_idx].to(x_mv.dtype).to(x_mv.device)  # [16, 16]
        x_mv_g = x_mv_g @ M_g.T  # [..., C_mv, 16]

        # Forward pass on transformed input
        y_mv_g, y_s_g = model_fn(x_mv_g, x_s_g)

        # Forward pass on original input, then apply g
        y_mv, y_s = model_fn(x_mv, x_s)
        y_mv_out, y_s_out = apply_group_action_to_tokens(backend, g_idx, y_mv, y_s)
        y_mv_out = y_mv_out @ M_g.T

        # Compute error
        err = (y_mv_g - y_mv_out).pow(2).mean()
        if y_s_g is not None and y_s_out is not None:
            err = err + (y_s_g - y_s_out).pow(2).mean()
        total_err = total_err + err

    return total_err / num_samples

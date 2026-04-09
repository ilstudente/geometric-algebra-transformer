"""Equivariance measurement utilities.

Provides functions to measure:
  - Global equivariance error of a model
  - Per-layer equivariance error (via forward hooks)
  - Parameter count
  - Memory usage estimate
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.pga.actions import (
    apply_group_action_to_tokens,
    equivariance_error,
)


def compute_equivariance_error(
    model: nn.Module,
    backend: FiniteSymmetryBackend,
    x_mv: Tensor,
    x_s: Tensor | None = None,
    num_samples: int = 8,
) -> float:
    """Compute the average finite-group equivariance error over num_samples group elements.

    E = mean_{g} || f(T_g x) - T_g f(x) ||^2 / || f(x) ||^2

    Parameters
    ----------
    model : nn.Module
        Must have signature (x_mv, x_s) -> (y_mv, y_s).
    backend : FiniteSymmetryBackend
    x_mv : Tensor [B, X, C_mv, 16]
    x_s : Tensor [B, X, C_s] or None
    num_samples : int

    Returns
    -------
    float
    """
    model.eval()
    with torch.no_grad():
        def model_fn(mv, s):
            return model(mv, s)
        err = equivariance_error(backend, model_fn, x_mv, x_s, num_samples)
    return float(err.item())


def compute_per_layer_equivariance_error(
    model: nn.Module,
    backend: FiniteSymmetryBackend,
    x_mv: Tensor,
    x_s: Tensor | None = None,
    num_samples: int = 4,
) -> Dict[str, float]:
    """Measure equivariance error at each named module's output via hooks.

    Parameters
    ----------
    model : nn.Module
    backend : FiniteSymmetryBackend
    x_mv : Tensor [B, X, C_mv, 16]
    x_s : Tensor or None
    num_samples : int

    Returns
    -------
    dict mapping module name -> equivariance error
    """
    # Sample group elements
    G = backend.size()
    g_indices = torch.randperm(G)[:num_samples].tolist()
    M_mats = [
        backend.mv_rep[g].to(x_mv.dtype).to(x_mv.device)
        for g in g_indices
    ]

    errors = {}
    activations_original: Dict[str, Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out_mv = output[0]
            else:
                out_mv = output
            if out_mv is not None and isinstance(out_mv, Tensor) and out_mv.dim() >= 3:
                activations_original[name] = out_mv.detach().clone()
        return hook

    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                h = module.register_forward_hook(make_hook(name))
                handles.append(h)

        # Forward on original
        _ = model(x_mv, x_s)

        # For each group element, forward on transformed input and compare
        for g_idx, M_g in zip(g_indices, M_mats):
            acts_transformed: Dict[str, Tensor] = {}

            def make_hook_t(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out_mv = output[0]
                    else:
                        out_mv = output
                    if out_mv is not None and isinstance(out_mv, Tensor) and out_mv.dim() >= 3:
                        acts_transformed[name] = out_mv.detach().clone()
                return hook

            handles_t = []
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:
                    h = module.register_forward_hook(make_hook_t(name))
                    handles_t.append(h)

            # Transform input
            x_mv_g, x_s_g = apply_group_action_to_tokens(backend, g_idx, x_mv, x_s)
            x_mv_g = x_mv_g @ M_g.T

            _ = model(x_mv_g, x_s_g)

            for h in handles_t:
                h.remove()

            # Compare activations
            for name in activations_original:
                if name in acts_transformed:
                    orig = activations_original[name]
                    trans = acts_transformed[name]
                    if orig.shape == trans.shape:
                        # Equivariance error: ||trans - orig||^2 / ||orig||^2
                        err = (trans - orig).pow(2).mean().item()
                        errors[name] = errors.get(name, 0.0) + err / num_samples

    for h in handles:
        h.remove()

    return errors


def compute_scalar_equivariance_error(
    scalar_fn,
    backend: FiniteSymmetryBackend,
    x_s: Tensor,
    num_samples: int = 8,
) -> float:
    """Measure the finite-group equivariance error of a scalar-only function.

    For a function f_s acting on scalar token sequences:

        E[g] = || f_s(T_g s) - T_g f_s(s) ||^2   averaged over g

    where T_g permutes the token axis (no multivector rotation).

    Parameters
    ----------
    scalar_fn : callable  x_s -> y_s
        A function mapping ``[..., X, C_s]`` to ``[..., X, C_s]``.
    backend : FiniteSymmetryBackend
    x_s : Tensor [..., X, C_s]
    num_samples : int

    Returns
    -------
    float
    """
    G = backend.size()
    indices = torch.randperm(G)[:num_samples]

    total_err = 0.0
    with torch.no_grad():
        for g_idx in indices.tolist():
            # Permute token axis: T_g s
            token_idx = torch.arange(G, device=x_s.device)
            perm = backend.action_on_group_tokens(g_idx, token_idx)
            x_s_g = x_s[..., perm, :]

            y_s_g = scalar_fn(x_s_g)         # f_s(T_g s)
            y_s = scalar_fn(x_s)             # f_s(s)
            y_s_out = y_s[..., perm, :]      # T_g f_s(s)

            err = (y_s_g - y_s_out).pow(2).mean()
            total_err = total_err + float(err.item())

    return total_err / num_samples


def compute_scalar_path_metrics(
    model: nn.Module,
    backend: FiniteSymmetryBackend,
    x_s: Tensor,
    num_samples: int = 8,
) -> dict:
    """Compute scalar-path equivariance error and parameter counts.

    Walks the model looking for ScalarGMLinear modules and aggregates
    their error_norm and displacement_proxy values.

    Parameters
    ----------
    model : nn.Module
    backend : FiniteSymmetryBackend
    x_s : Tensor [..., X, C_s]
    num_samples : int

    Returns
    -------
    dict with keys:
        scalar_equiv_error, total_error_norm, total_displacement,
        gm_scalar_params, total_params, gm_fraction
    """
    from finite_subgroup_gatr.layers.scalar_gm_linear import ScalarGMLinear

    total_error_norm = 0.0
    total_displacement = 0.0
    gm_scalar_params = 0

    for module in model.modules():
        if isinstance(module, ScalarGMLinear):
            total_error_norm += float(module.error_norm().item())
            total_displacement += float(module.displacement_proxy().item())
            gm_scalar_params += sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

    total_params = count_parameters(model)
    gm_fraction = gm_scalar_params / max(total_params, 1)

    # Scalar equivariance error: pass x_s through the full model's scalar path
    # We use a wrapper that extracts just the scalar output
    def scalar_fn(s):
        model.eval()
        with torch.no_grad():
            # Build a dummy x_mv matching the token dim
            X = s.shape[-2]
            dummy_mv = torch.zeros(*s.shape[:-2], X, 1, 16, device=s.device, dtype=s.dtype)
            _, out_s = model(dummy_mv, s)
        return out_s if out_s is not None else s

    try:
        scalar_equiv_error = compute_scalar_equivariance_error(
            scalar_fn, backend, x_s, num_samples
        )
    except Exception:
        scalar_equiv_error = float("nan")

    return {
        "scalar_equiv_error": scalar_equiv_error,
        "total_error_norm": total_error_norm,
        "total_displacement": total_displacement,
        "gm_scalar_params": gm_scalar_params,
        "total_params": total_params,
        "gm_fraction": gm_fraction,
    }


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_mb(model: nn.Module, x_mv: Tensor, x_s: Tensor | None = None) -> float:
    """Rough estimate of peak memory in MB during a forward pass."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    input_bytes = x_mv.numel() * x_mv.element_size()
    if x_s is not None:
        input_bytes += x_s.numel() * x_s.element_size()
    # Rough factor for activations: ~3x parameter memory for typical networks
    total_bytes = param_bytes + 3 * input_bytes
    return total_bytes / (1024 ** 2)

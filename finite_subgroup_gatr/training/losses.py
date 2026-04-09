"""Loss functions for FiniteSubgroupGATr training.

Total loss:
  L = task_loss
    + lambda_eq   * equivariance_loss
    + lambda_error * error_norm_loss
    + lambda_disp  * displacement_proxy_loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.pga.actions import equivariance_error


class FiniteSubgroupLoss(nn.Module):
    """Composite loss for finite-subgroup equivariant training.

    Parameters
    ----------
    task_loss_fn : nn.Module or callable
        The primary task loss (e.g. MSE, cross-entropy).
    lambda_eq : float
        Weight for finite-group equivariance regularization.
    lambda_error : float
        Weight for residual error norm regularization.
    lambda_disp : float
        Weight for displacement proxy regularization.
    eq_num_samples : int
        Number of group elements to sample for equivariance loss.
    """

    def __init__(
        self,
        task_loss_fn,
        lambda_eq: float = 0.0,
        lambda_error: float = 0.0,
        lambda_disp: float = 0.0,
        eq_num_samples: int = 8,
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.lambda_eq = lambda_eq
        self.lambda_error = lambda_error
        self.lambda_disp = lambda_disp
        self.eq_num_samples = eq_num_samples

    def forward(
        self,
        pred,
        target,
        model: nn.Module | None = None,
        x_mv: Tensor | None = None,
        x_s: Tensor | None = None,
        backend: FiniteSymmetryBackend | None = None,
    ) -> tuple[Tensor, dict]:
        """Compute combined loss.

        Parameters
        ----------
        pred : model prediction
        target : ground truth
        model : nn.Module (needed for equivariance and error terms)
        x_mv : Tensor  (needed for equivariance term)
        x_s : Tensor or None
        backend : FiniteSymmetryBackend (needed for equivariance term)

        Returns
        -------
        total_loss : Tensor scalar
        metrics : dict of named loss components
        """
        metrics = {}

        # Task loss
        task_loss = self.task_loss_fn(pred, target)
        metrics["task_loss"] = task_loss.detach().item()
        total = task_loss

        # Equivariance loss
        if self.lambda_eq > 0 and model is not None and x_mv is not None and backend is not None:
            def model_fn(mv, s):
                return model(mv, s)
            eq_loss = equivariance_error(backend, model_fn, x_mv, x_s, self.eq_num_samples)
            metrics["equivariance_loss"] = eq_loss.detach().item()
            total = total + self.lambda_eq * eq_loss

        # Error norm loss
        if self.lambda_error > 0 and model is not None:
            error_norm = _collect_error_norm(model)
            metrics["error_norm_loss"] = error_norm.detach().item()
            total = total + self.lambda_error * error_norm

        # Displacement proxy loss
        if self.lambda_disp > 0 and model is not None:
            disp = _collect_displacement_proxy(model)
            metrics["displacement_proxy_loss"] = disp.detach().item()
            total = total + self.lambda_disp * disp

        metrics["total_loss"] = total.detach().item()
        return total, metrics


def _collect_error_norm(model: nn.Module) -> Tensor:
    """Collect total residual error norm from all GMTokenMixer modules."""
    total = torch.zeros(1, device=next(model.parameters()).device)
    for module in model.modules():
        if hasattr(module, "error_norm") and callable(module.error_norm):
            n = module.error_norm()
            if n.device != total.device:
                n = n.to(total.device)
            total = total + n
    return total


def _collect_displacement_proxy(model: nn.Module) -> Tensor:
    """Collect displacement proxy from all GMTokenMixer modules."""
    total = torch.zeros(1, device=next(model.parameters()).device)
    for module in model.modules():
        if hasattr(module, "displacement_proxy") and callable(module.displacement_proxy):
            d = module.displacement_proxy()
            if d.device != total.device:
                d = d.to(total.device)
            total = total + d
    return total

"""Approximate equivariance structures for GM token mixing.

Implements three modes:
  - "none"           : exact equivariant GM weights
  - "low_rank"       : W = W_GM + U V^T  (low-rank residual)
  - "dense_penalized": W = W_GM + E       (dense residual with regularization)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ApproxLinear(nn.Module):
    """A channel-mixing linear layer with optional approximate-equivariance residual.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    error_mode : {"none", "low_rank", "dense_penalized"}
    error_rank : int
        Rank of the low-rank residual (used when error_mode == "low_rank").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        error_mode: str = "none",
        error_rank: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.error_mode = error_mode

        # Exact GM weight (shared across all elements in a group orbit)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))

        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Residual error term
        self.error_U: nn.Parameter | None = None
        self.error_V: nn.Parameter | None = None
        self.error_E: nn.Parameter | None = None

        if error_mode == "low_rank":
            self.error_U = nn.Parameter(torch.zeros(out_channels, error_rank))
            self.error_V = nn.Parameter(torch.zeros(in_channels, error_rank))
        elif error_mode == "dense_penalized":
            self.error_E = nn.Parameter(torch.zeros(out_channels, in_channels))

    def effective_weight(self) -> Tensor:
        """Return the effective weight matrix W_GM + error."""
        W = self.weight
        if self.error_mode == "low_rank" and self.error_U is not None:
            W = W + self.error_U @ self.error_V.T
        elif self.error_mode == "dense_penalized" and self.error_E is not None:
            W = W + self.error_E
        return W

    def error_norm(self) -> Tensor:
        """Return the Frobenius norm of the residual error."""
        if self.error_mode == "low_rank" and self.error_U is not None:
            E = self.error_U @ self.error_V.T
            return E.pow(2).sum().sqrt()
        elif self.error_mode == "dense_penalized" and self.error_E is not None:
            return self.error_E.pow(2).sum().sqrt()
        return torch.zeros(1, device=self.weight.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor [..., in_channels]

        Returns
        -------
        Tensor [..., out_channels]
        """
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)


class GMMixingWeights(nn.Module):
    """Weight bank for GM token mixing: one ApproxLinear per neighborhood element.

    Parameters
    ----------
    K : int
        Neighborhood size.
    in_channels : int
    out_channels : int
    error_mode : str
    error_rank : int
    """

    def __init__(
        self,
        K: int,
        in_channels: int,
        out_channels: int,
        error_mode: str = "none",
        error_rank: int = 4,
    ):
        super().__init__()
        self.weights = nn.ModuleList([
            ApproxLinear(in_channels, out_channels, error_mode, error_rank)
            for _ in range(K)
        ])

    def error_norm(self) -> Tensor:
        """Total Frobenius norm of all residual errors."""
        total = sum(w.error_norm() for w in self.weights)
        return total

    def displacement_proxy(self) -> Tensor:
        """Proxy for displacement from exact GM form.

        Measures the variance of weights across the neighborhood elements.
        For exact GM, all weights should be related by the group structure;
        deviations from this indicate approximate equivariance.
        """
        if len(self.weights) <= 1:
            return torch.zeros(1, device=self.weights[0].weight.device)
        weights_stack = torch.stack([w.effective_weight() for w in self.weights], dim=0)
        mean_w = weights_stack.mean(dim=0, keepdim=True)
        return (weights_stack - mean_w).pow(2).sum().sqrt()

# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.

"""Generalized group matrix convolution layers."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence, TypeVar

import torch
import torch.nn as nn

GroupElement = TypeVar("GroupElement")
SpaceElement = TypeVar("SpaceElement")


class GMConvGeneralBase(nn.Module):
    """Generalized Group Matrix Convolution base for finite actions G ⤳ X.

    Parameters
    ----------
    group_elements:
        Iterable of group elements in :math:`G`.
    space_elements:
        Iterable of space elements in :math:`X`.
    group_action:
        Callable implementing the action ``(g, x) ↦ g·x``.
    group_inverse_func:
        Callable returning ``g⁻¹`` for a group element ``g``.
    nbr_elements:
        Sequence of group elements that define the convolution kernel support.
    out_channels:
        Number of output channels.
    error:
        When ``True`` enables the approximate equivariance error modulation.
    in_channels:
        Number of input channels.
    """

    def __init__(
        self,
        group_elements: Iterable[GroupElement],
        space_elements: Iterable[SpaceElement],
        group_action: Callable[[GroupElement, SpaceElement], SpaceElement],
        group_inverse_func: Callable[[GroupElement], GroupElement],
        nbr_elements: Sequence[GroupElement],
        out_channels: int,
        *,
        error: bool = False,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.group_elements = list(group_elements)
        self.space_elements = list(space_elements)
        self.group_action = group_action
        self.group_inverse_func = group_inverse_func
        self.nbr_elements = list(nbr_elements)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.error = error

        self.element_to_idx = {x: i for i, x in enumerate(self.space_elements)}

        index_matrix = self._build_index_matrix()
        self.register_buffer("index_matrix", index_matrix.long(), persistent=False)

        self.bias = nn.Parameter(torch.zeros(out_channels))

        # weight: (out, in, K)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, len(self.nbr_elements))
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.error:
            # err_vector modulates the last (space) axis
            self.err_vector = nn.Parameter(torch.empty(1, len(self.space_elements)))
            nn.init.kaiming_uniform_(self.err_vector, a=math.sqrt(5))

    def _build_index_matrix(self) -> torch.Tensor:
        N = len(self.space_elements)
        K = len(self.nbr_elements)
        idx = torch.empty(K, N, dtype=torch.long)
        for k, h in enumerate(self.nbr_elements):
            h_inv = self.group_inverse_func(h)
            for i, x in enumerate(self.space_elements):
                src = self.group_action(h_inv, x)  # y = h^{-1}·x
                idx[k, i] = self.element_to_idx[src]  # y index
        return idx

    def _maybe_apply_error(self, adj_input: torch.Tensor) -> torch.Tensor:
        """Apply the approximate equivariance error modulation if enabled."""

        if not self.error:
            return adj_input
        with torch.no_grad():
            n = self.err_vector.norm().clamp_min(1e-12)
            self.err_vector.div_(n)
            wnorm = self.weight.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            self.weight.div_(wnorm)
        # ``err_vector`` has shape (1, N); reshape to broadcast over (B, Cin, K, N)
        return adj_input * self.err_vector.view(1, 1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the generalized group matrix convolution."""

        if x.dim() == 4 and x.size(-1) == len(self.space_elements):
            # (B, Cin, 1, N) -> (B, Cin, N)
            x = x[:, :, 0, :]
        if x.dim() != 3 or x.size(-1) != len(self.space_elements):
            raise ValueError(f"x must be (B, Cin, N) with N={len(self.space_elements)}")

        B, Cin, N = x.shape
        if Cin != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: got {Cin}, expected {self.in_channels}"
            )

        # Gather ψ(h^{-1}·x): index_matrix is (K, N)
        # Fancy indexing yields (B, Cin, K, N)
        adj = x[:, :, self.index_matrix]

        adj = self._maybe_apply_error(adj)  # (B, Cin, K, N)

        # Weighted sum over (in, K) -> (B, Cout, N)
        out = torch.einsum("oik,bikn->bon", self.weight, adj)

        return out + self.bias[None, :, None]

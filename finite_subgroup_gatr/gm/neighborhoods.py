"""Neighborhood utilities for GM convolution.

Provides functions to retrieve and cache neighborhoods at various radii,
and to compute the Cayley graph structure of a finite group backend.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend


def get_neighborhood(
    backend: FiniteSymmetryBackend,
    radius: int,
) -> Tensor:
    """Return the word-metric neighborhood of radius r (including identity).

    Parameters
    ----------
    backend : FiniteSymmetryBackend
    radius : int

    Returns
    -------
    LongTensor [K]
    """
    return backend.neighborhood(radius)


def neighborhood_contains_identity(
    backend: FiniteSymmetryBackend,
    radius: int,
) -> bool:
    """Check that the neighborhood contains the identity."""
    nbr = get_neighborhood(backend, radius)
    identity = backend.identity()
    return identity in nbr.tolist()


def cayley_distance_matrix(backend: FiniteSymmetryBackend) -> Tensor:
    """Compute the full pairwise Cayley (word-metric) distance matrix.

    Returns
    -------
    LongTensor [G, G]  where entry [i, j] = word_dist(i, j)
    """
    G = backend.size()
    dist_matrix = torch.zeros(G, G, dtype=torch.long)
    for g in range(G):
        nbr = backend.neighborhood(G)  # full group
        # dist from g to each other element
        for r in range(G):
            ng = get_neighborhood(backend, r)
            # Compute elements reachable in r steps from g
            # distance from g to h = distance from identity to g^{-1}*h
            pass  # simplified: distance from identity to g is r iff g in N_r \ N_{r-1}

    # Simpler O(G^2) computation using mul table
    inv = backend.inv_table
    for i in range(G):
        for j in range(G):
            # distance(i, j) = distance(identity, i^{-1}*j)
            h = int(backend.mul_table[int(inv[i]), j])
            # find minimum r s.t. h in N_r
            for r in range(G):
                nbr = get_neighborhood(backend, r)
                if h in nbr.tolist():
                    dist_matrix[i, j] = r
                    break

    return dist_matrix

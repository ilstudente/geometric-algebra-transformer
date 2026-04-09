"""Abstract base class for finite symmetry group backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

import torch


class FiniteSymmetryBackend(ABC):
    """Protocol for finite discrete group backends.

    Subclasses provide:
      - Group enumeration and algebraic structure (multiplication, inverse)
      - Action tables for permuting token indices
      - Actions on 3D points and PGA multivectors (16D)
      - Neighborhood sets for GM convolution

    All group elements are represented as integer indices into a canonical list.
    """

    # ------------------------------------------------------------------ #
    # Core group structure                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def elements(self) -> List[Any]:
        """Return a list of all group elements (canonical representation)."""

    @abstractmethod
    def size(self) -> int:
        """Return |G_f|."""

    @abstractmethod
    def identity(self) -> int:
        """Return the index of the identity element."""

    @abstractmethod
    def multiply(self, a: int, b: int) -> int:
        """Return the index of g_a * g_b."""

    @abstractmethod
    def inverse(self, a: int) -> int:
        """Return the index of g_a^{-1}."""

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def action_on_group_tokens(self, g: int, x: torch.Tensor) -> torch.Tensor:
        """Apply group element g to a batch of token indices.

        Parameters
        ----------
        g : int
            Group element index.
        x : LongTensor [..., X]
            Token indices in [0, |G_f|).

        Returns
        -------
        LongTensor [..., X]
            Permuted token indices: result[i] = index of g * element(x[i]).
        """

    @abstractmethod
    def action_on_points(self, g: int, pts: torch.Tensor) -> torch.Tensor:
        """Apply group element g to 3D points.

        Parameters
        ----------
        g : int
            Group element index.
        pts : Tensor [..., 3]

        Returns
        -------
        Tensor [..., 3]
        """

    @abstractmethod
    def action_on_multivectors(self, g: int, mv: torch.Tensor) -> torch.Tensor:
        """Apply group element g to PGA multivectors via the precomputed 16x16 matrix.

        Parameters
        ----------
        g : int
            Group element index.
        mv : Tensor [..., 16]

        Returns
        -------
        Tensor [..., 16]
        """

    # ------------------------------------------------------------------ #
    # Neighborhood / subgroup / coset                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def neighborhood(self, radius: int) -> torch.Tensor:
        """Return the word-metric ball of radius r around the identity.

        Parameters
        ----------
        radius : int

        Returns
        -------
        LongTensor [K]
            Indices of elements in the neighborhood.
        """

    @abstractmethod
    def subgroup(self, name: str) -> "FiniteSymmetryBackend":
        """Return a named subgroup as a new backend."""

    @abstractmethod
    def coset_partition(
        self, subgroup: "FiniteSymmetryBackend"
    ) -> List[List[int]]:
        """Partition G_f into left cosets of the given subgroup.

        Returns
        -------
        list of list of int
            Each inner list contains the indices of one coset.
        """

    # ------------------------------------------------------------------ #
    # Convenience: stored tables                                            #
    # ------------------------------------------------------------------ #

    @property
    def mul_table(self) -> torch.Tensor:
        """LongTensor [G, G] - multiplication table."""
        raise NotImplementedError

    @property
    def inv_table(self) -> torch.Tensor:
        """LongTensor [G] - inverse table."""
        raise NotImplementedError

    @property
    def left_action_table(self) -> torch.Tensor:
        """LongTensor [G, X] - left action on token domain.

        left_action_table[g, x] = index of g * element(x).
        """
        raise NotImplementedError

    @property
    def mv_rep(self) -> torch.Tensor:
        """FloatTensor [G, 16, 16] - multivector representation matrices."""
        raise NotImplementedError

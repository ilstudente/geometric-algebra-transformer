"""Grid x rotation backend: G_f = T_grid ⋊ H_R.

The token domain is a product of a bounded integer grid and a finite rotation group.
This is the closest discrete Euclidean surrogate for E(3).

Group elements are pairs (t, r) where:
  - t is a grid point in Z^3 / (N_x x N_y x N_z)
  - r is a rotation in H_R (a RotationOnlyBackend)

Group law: (t1, r1) * (t2, r2) = (t1 + r1 * t2, r1 * r2)
(semidirect product with H_R acting on the translation lattice)
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend
from finite_subgroup_gatr.backends.rotation_backend import RotationOnlyBackend


class GridRotationBackend(FiniteSymmetryBackend):
    """Finite translation grid x finite rotation group.

    Parameters
    ----------
    grid_shape : tuple of int (N_x, N_y, N_z)
        Size of the grid along each axis.  Grid points are in {0,...,N_x-1}^3.
    rotation_backend : RotationOnlyBackend
        Finite rotation group H_R.
    grid_spacing : float
        Physical spacing between grid points (for point action).
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        rotation_backend: RotationOnlyBackend,
        grid_spacing: float = 1.0,
    ):
        self.grid_shape = grid_shape
        self.rot = rotation_backend
        self.grid_spacing = grid_spacing

        Nx, Ny, Nz = grid_shape
        R = rotation_backend.size()

        # Enumerate grid points
        gx = np.arange(Nx)
        gy = np.arange(Ny)
        gz = np.arange(Nz)
        grid_points = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
        self._grid_points = grid_points  # [T, 3]
        T = len(grid_points)
        self._T = T
        self._R = R

        # Flat index: element (t_idx, r_idx) -> t_idx * R + r_idx
        G = T * R
        self._G = G

        # Build multiplication table
        mul_table = np.zeros((G, G), dtype=np.int64)
        inv_table = np.zeros(G, dtype=np.int64)

        rot_mats = rotation_backend.rotation_matrices  # [R, 3, 3]
        rot_mul = rotation_backend.mul_table.numpy()   # [R, R]
        rot_inv = rotation_backend.inv_table.numpy()   # [R]

        for i in range(G):
            t1_idx, r1_idx = divmod(i, R)
            t1 = grid_points[t1_idx]
            for j in range(G):
                t2_idx, r2_idx = divmod(j, R)
                t2 = grid_points[t2_idx]

                # Compute t1 + r1 * t2 (mod grid)
                t2_rot = rot_mats[r1_idx] @ t2.astype(float)
                t_new = (np.round(t1.astype(float) + t2_rot) % np.array([Nx, Ny, Nz])).astype(int)

                # Find t_new in grid_points
                t_new_idx = self._find_grid_point(grid_points, t_new)

                r_new_idx = int(rot_mul[r1_idx, r2_idx])
                mul_table[i, j] = t_new_idx * R + r_new_idx

        for i in range(G):
            # Find inverse: (t, r)^{-1} = (-r^{-1}*t, r^{-1})
            t_idx, r_idx = divmod(i, R)
            t = grid_points[t_idx]
            r_inv_idx = int(rot_inv[r_idx])
            t_inv = (-rot_mats[r_inv_idx] @ t.astype(float) % np.array([Nx, Ny, Nz])).astype(int)
            t_inv_idx = self._find_grid_point(grid_points, np.round(t_inv).astype(int))
            inv_table[i] = t_inv_idx * R + r_inv_idx

        self._mul_table_np = mul_table
        self._mul_table_t = torch.from_numpy(mul_table)
        self._inv_table_t = torch.from_numpy(inv_table)

        # MV representation matrices: (t, r) acts on PGA multivectors via r only
        # (translations act differently—as motors—but for pure rotations we use the rotation part)
        rot_mv_rep = rotation_backend.mv_rep.numpy()  # [R, 16, 16]
        mv_rep = np.zeros((G, 16, 16), dtype=np.float32)
        for i in range(G):
            _, r_idx = divmod(i, R)
            mv_rep[i] = rot_mv_rep[r_idx]
        self._mv_rep_t = torch.from_numpy(mv_rep)

        # Neighborhoods: BFS on the product group
        self._neighborhoods = self._compute_neighborhoods(mul_table)

    @staticmethod
    def _find_grid_point(grid_points: np.ndarray, pt: np.ndarray) -> int:
        diffs = np.abs(grid_points - pt[None]).sum(axis=1)
        idx = int(np.argmin(diffs))
        assert diffs[idx] < 0.5, f"Grid point {pt} not found"
        return idx

    def _compute_neighborhoods(self, mul_table: np.ndarray) -> List[np.ndarray]:
        """BFS word-metric neighborhoods using minimal generators."""
        from collections import deque

        G = self._G
        R = self._R
        Nx, Ny, Nz = self.grid_shape

        # Generators: unit translation in each direction + rotation generators
        generators = []
        grid_points = self._grid_points

        # Unit translations (if grid > 1)
        for axis in range(3):
            shift = np.zeros(3, dtype=int)
            shift[axis] = 1
            t_idx = self._find_grid_point(grid_points, shift % np.array([Nx, Ny, Nz]))
            generators.append(t_idx * R + 0)  # translation with identity rotation

        # Rotation generators (index 1, 2 in rotation group)
        for r_gen in self.rot._get_generators():
            generators.append(0 * R + r_gen)  # identity translation + rotation gen

        # Add inverses
        inv_gens = [int(self._inv_table_t[g]) for g in generators]
        all_gens = list(set(generators + inv_gens))

        dist = np.full(G, -1, dtype=np.int64)
        dist[0] = 0
        queue = deque([0])
        while queue:
            x = queue.popleft()
            for g in all_gens:
                y = mul_table[g, x]
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    queue.append(int(y))

        max_dist = int(dist.max())
        return [np.where(dist <= r)[0] for r in range(max_dist + 1)]

    # ---- FiniteSymmetryBackend interface ----

    def elements(self) -> List[Tuple[np.ndarray, int]]:
        elems = []
        for i in range(self._G):
            t_idx, r_idx = divmod(i, self._R)
            elems.append((self._grid_points[t_idx], r_idx))
        return elems

    def size(self) -> int:
        return self._G

    def identity(self) -> int:
        return 0

    def multiply(self, a: int, b: int) -> int:
        return int(self._mul_table_t[a, b])

    def inverse(self, a: int) -> int:
        return int(self._inv_table_t[a])

    def action_on_group_tokens(self, g: int, x: torch.Tensor) -> torch.Tensor:
        return self._mul_table_t[g][x]

    def action_on_points(self, g: int, pts: torch.Tensor) -> torch.Tensor:
        t_idx, r_idx = divmod(g, self._R)
        R_mat = torch.from_numpy(self.rot.rotation_matrices[r_idx]).to(pts.dtype).to(pts.device)
        t = torch.from_numpy(self._grid_points[t_idx] * self.grid_spacing).to(pts.dtype).to(pts.device)
        return pts @ R_mat.T + t

    def action_on_multivectors(self, g: int, mv: torch.Tensor) -> torch.Tensor:
        M = self._mv_rep_t[g].to(mv.dtype).to(mv.device)
        return mv @ M.T

    def neighborhood(self, radius: int) -> torch.Tensor:
        r = min(radius, len(self._neighborhoods) - 1)
        return torch.from_numpy(self._neighborhoods[r].copy()).long()

    def subgroup(self, name: str) -> "FiniteSymmetryBackend":
        if name == "rotation":
            return self.rot
        raise ValueError(f"Unknown subgroup '{name}'")

    def coset_partition(self, subgroup: "FiniteSymmetryBackend") -> List[List[int]]:
        # Partition by left cosets
        sub_size = subgroup.size()
        sub_indices = set()

        for i in range(sub_size):
            for j in range(self._G):
                # Check if element j corresponds to subgroup element i
                t_idx, r_idx = divmod(j, self._R)
                t = self._grid_points[t_idx]
                if np.all(t == 0) and r_idx < self.rot.size():
                    sub_indices.add(j)
                if len(sub_indices) >= sub_size:
                    break

        seen = set()
        cosets = []
        for g in range(self._G):
            if g in seen:
                continue
            coset = []
            for h in sub_indices:
                gh = int(self._mul_table_t[g, h])
                coset.append(gh)
                seen.add(gh)
            cosets.append(coset)
        return cosets

    @property
    def mul_table(self) -> torch.Tensor:
        return self._mul_table_t

    @property
    def inv_table(self) -> torch.Tensor:
        return self._inv_table_t

    @property
    def left_action_table(self) -> torch.Tensor:
        return self._mul_table_t

    @property
    def mv_rep(self) -> torch.Tensor:
        return self._mv_rep_t

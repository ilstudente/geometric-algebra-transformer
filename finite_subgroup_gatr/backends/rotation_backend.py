"""Finite rotation group backends: cyclic, tetrahedral, octahedral, icosahedral.

Each group is represented by its 3x3 rotation matrices.  Multiplication and
inverse tables are built once at construction time, as are the 16x16 PGA
multivector action matrices (computed numerically via the versor product).

Token domain: G_f acts on itself by left multiplication (regular representation).
"""

from __future__ import annotations

from collections import deque
from itertools import permutations
from typing import Any, List, Optional

import numpy as np
import torch

from finite_subgroup_gatr.backends.base import FiniteSymmetryBackend


# ---------------------------------------------------------------------------
# Quaternion / rotation matrix utilities
# ---------------------------------------------------------------------------

def _rotation_matrix_to_quaternion_np(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w] (Hamilton)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _rotor_from_quaternion_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Build a 16-component PGA rotor from a quaternion [x, y, z, w].

    Uses the GATr convention (gatr/interface/rotation.py):
      multivector[0]  = w             (scalar)
      multivector[8]  = -z            (-e12)
      multivector[9]  = y             (e13)
      multivector[10] = -x            (-e23)
    """
    qx, qy, qz, qw = q_xyzw
    r = np.zeros(16, dtype=np.float64)
    r[0] = qw
    r[8] = -qz
    r[9] = qy
    r[10] = -qx
    return r


def _reverse_rotor_np(r: np.ndarray) -> np.ndarray:
    """Reverse operation: negate indices 5..14 (bivectors and trivectors)."""
    rev = r.copy()
    rev[5:15] = -rev[5:15]
    return rev


# Precompute GP table once (using the same data as GATr primitives)
_GP_TABLE: Optional[np.ndarray] = None


def _get_gp_table() -> np.ndarray:
    """Load the geometric product kernel from GATr's data files."""
    global _GP_TABLE
    if _GP_TABLE is None:
        import os
        data_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "gatr", "primitives", "data", "geometric_product.pt",
        )
        data_path = os.path.abspath(data_path)
        sparse_tensor = torch.load(data_path, map_location="cpu", weights_only=False)
        _GP_TABLE = sparse_tensor.to_dense().numpy().astype(np.float64)
    return _GP_TABLE


def _gp_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute geometric product of two multivectors using numpy."""
    gp = _get_gp_table()
    return np.einsum("ijk,j,k->i", gp, x, y)


def _compute_mv_action_matrix(R: np.ndarray) -> np.ndarray:
    """Compute the 16x16 PGA multivector action matrix for rotation R.

    Uses the versor (sandwich) product: x -> r x r_rev,
    where r is the rotor corresponding to R.
    """
    q = _rotation_matrix_to_quaternion_np(R)
    r = _rotor_from_quaternion_np(q)
    r_rev = _reverse_rotor_np(r)

    mat = np.zeros((16, 16), dtype=np.float64)
    for j in range(16):
        e_j = np.zeros(16, dtype=np.float64)
        e_j[j] = 1.0
        mat[:, j] = _gp_np(_gp_np(r, e_j), r_rev)

    return mat


def _find_matching_rotation(rotations: np.ndarray, R: np.ndarray, tol: float = 1e-6) -> int:
    """Find the index of R in a list of rotation matrices."""
    diffs = np.abs(rotations - R[None]).sum(axis=(1, 2))
    idx = int(np.argmin(diffs))
    if diffs[idx] > tol:
        raise ValueError(f"Could not find rotation in list (min diff = {diffs[idx]:.2e})")
    return idx


def _build_tables(rotations: np.ndarray):
    """Build multiplication table, inverse table, and MV representation."""
    G = len(rotations)

    mul_table = np.zeros((G, G), dtype=np.int64)
    for i in range(G):
        for j in range(G):
            R_ij = rotations[i] @ rotations[j]
            mul_table[i, j] = _find_matching_rotation(rotations, R_ij)

    inv_table = np.zeros(G, dtype=np.int64)
    for i in range(G):
        inv_table[i] = _find_matching_rotation(rotations, rotations[i].T)

    mv_rep = np.stack([_compute_mv_action_matrix(R) for R in rotations], axis=0)

    return (
        torch.from_numpy(mul_table),
        torch.from_numpy(inv_table),
        torch.from_numpy(mv_rep).float(),
    )


def _word_metric_neighborhoods(mul_table: np.ndarray, generators: List[int]) -> List[np.ndarray]:
    """BFS to compute word-metric distance from identity for all group elements.

    Returns list indexed by radius: neighborhoods[r] contains indices at distance <= r.

    The generating set is closed under inversion automatically.
    If not all elements are reachable (generators don't generate the full group),
    unreachable elements get distance G (large sentinel).
    """
    G = len(mul_table)
    dist = np.full(G, -1, dtype=np.int64)
    dist[0] = 0  # identity is always index 0
    queue = deque([0])

    # Build gen_set: generators + their inverses, deduplicated
    gen_set_set = set(generators)
    for g in generators:
        # Find h such that mul_table[g, h] = 0 (identity)
        inv_g = int(np.argmin(np.abs(mul_table[g] - 0)))
        # More robust: find exact
        matches = np.where(mul_table[g] == 0)[0]
        if len(matches) > 0:
            gen_set_set.add(int(matches[0]))
    gen_set = list(gen_set_set)

    while queue:
        x = queue.popleft()
        for g in gen_set:
            y = int(mul_table[g, x])
            if dist[y] == -1:
                dist[y] = dist[x] + 1
                queue.append(y)

    # Handle unreachable elements: assign distance G (so they appear only in large neighborhoods)
    dist[dist == -1] = G

    max_dist = int(dist[dist < G].max()) if (dist < G).any() else 0
    neighborhoods = []
    for r in range(max_dist + 1):
        neighborhoods.append(np.where(dist <= r)[0])

    # Add final neighborhood covering all elements
    if len(neighborhoods) == 0 or len(neighborhoods[-1]) < G:
        neighborhoods.append(np.arange(G))

    return neighborhoods


# ---------------------------------------------------------------------------
# Base class for rotation-only backends
# ---------------------------------------------------------------------------

class RotationOnlyBackend(FiniteSymmetryBackend):
    """Base class for finite rotation group backends.

    Subclasses must implement _enumerate_rotations() and _get_generators().
    Token domain = G_f (regular representation): X_f = G_f.
    """

    def __init__(self):
        rots = self._enumerate_rotations()
        # Ensure identity is first
        identity_idx = _find_matching_rotation(rots, np.eye(3))
        if identity_idx != 0:
            rots[[0, identity_idx]] = rots[[identity_idx, 0]]
        self._rotations = rots  # [G, 3, 3]

        mul_np = np.zeros((len(rots), len(rots)), dtype=np.int64)
        for i in range(len(rots)):
            for j in range(len(rots)):
                mul_np[i, j] = _find_matching_rotation(rots, rots[i] @ rots[j])

        self._mul_table, self._inv_table, self._mv_rep = _build_tables(rots)
        self._mul_np = mul_np

        gens = self._get_generators()
        self._neighborhoods = _word_metric_neighborhoods(mul_np, gens)

    # ---- abstract interface ----

    def _enumerate_rotations(self) -> np.ndarray:
        raise NotImplementedError

    def _get_generators(self) -> List[int]:
        raise NotImplementedError

    # ---- FiniteSymmetryBackend interface ----

    def elements(self) -> List[np.ndarray]:
        return [self._rotations[i] for i in range(len(self._rotations))]

    def size(self) -> int:
        return len(self._rotations)

    def identity(self) -> int:
        return 0

    def multiply(self, a: int, b: int) -> int:
        return int(self._mul_table[a, b])

    def inverse(self, a: int) -> int:
        return int(self._inv_table[a])

    def action_on_group_tokens(self, g: int, x: torch.Tensor) -> torch.Tensor:
        """Left action: returns table[g, x[i]] for each i."""
        return self._mul_table[g][x]

    def action_on_points(self, g: int, pts: torch.Tensor) -> torch.Tensor:
        R = torch.from_numpy(self._rotations[g]).to(pts.dtype).to(pts.device)
        return pts @ R.T

    def action_on_multivectors(self, g: int, mv: torch.Tensor) -> torch.Tensor:
        M = self._mv_rep[g].to(mv.dtype).to(mv.device)
        return mv @ M.T

    def neighborhood(self, radius: int) -> torch.Tensor:
        r = min(radius, len(self._neighborhoods) - 1)
        return torch.from_numpy(self._neighborhoods[r].copy()).long()

    def subgroup(self, name: str) -> "RotationOnlyBackend":
        raise NotImplementedError(f"Subgroup '{name}' not implemented for {type(self).__name__}")

    def coset_partition(self, subgroup: "FiniteSymmetryBackend") -> List[List[int]]:
        H_indices = set(subgroup.elements()) if isinstance(subgroup.elements()[0], int) else None
        # Find H as indices in this group
        h_rots = np.stack([subgroup.elements()[i] for i in range(subgroup.size())], axis=0)
        h_set = set()
        for h_rot in h_rots:
            h_set.add(_find_matching_rotation(self._rotations, h_rot))

        cosets = []
        seen = set()
        for g in range(self.size()):
            if g in seen:
                continue
            coset = []
            for h in h_set:
                gh = int(self._mul_table[g, h])
                coset.append(gh)
                seen.add(gh)
            cosets.append(coset)
        return cosets

    # ---- stored tables ----

    @property
    def mul_table(self) -> torch.Tensor:
        return self._mul_table

    @property
    def inv_table(self) -> torch.Tensor:
        return self._inv_table

    @property
    def left_action_table(self) -> torch.Tensor:
        # For regular representation, left_action_table = mul_table
        return self._mul_table

    @property
    def mv_rep(self) -> torch.Tensor:
        return self._mv_rep

    @property
    def rotation_matrices(self) -> np.ndarray:
        return self._rotations


# ---------------------------------------------------------------------------
# Cyclic group Z_n
# ---------------------------------------------------------------------------

class CyclicBackend(RotationOnlyBackend):
    """Cyclic group Z_n: n-fold rotations around the z-axis.

    Parameters
    ----------
    n : int
        Order of the group.
    """

    def __init__(self, n: int):
        self.n = n
        super().__init__()

    def _enumerate_rotations(self) -> np.ndarray:
        rotations = []
        for k in range(self.n):
            theta = 2 * np.pi * k / self.n
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1],
            ], dtype=np.float64)
            rotations.append(R)
        return np.stack(rotations, axis=0)

    def _get_generators(self) -> List[int]:
        # Generator: rotation by 2*pi/n (index 1)
        return [1]

    def subgroup(self, name: str) -> "CyclicBackend":
        if name.startswith("Z"):
            m = int(name[1:])
            assert self.n % m == 0, f"Z_{m} is not a subgroup of Z_{self.n}"
            return CyclicBackend(m)
        raise ValueError(f"Unknown subgroup name '{name}'")


# ---------------------------------------------------------------------------
# Tetrahedral group T (A_4, 12 elements)
# ---------------------------------------------------------------------------

def _enumerate_tetrahedral() -> np.ndarray:
    """Enumerate the 12 proper rotations of the tetrahedron.

    Generators: 3-fold rotation about (1,1,1)/sqrt(3) and 2-fold about z.
    """
    rotations = []

    def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = axis
        return np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
        ], dtype=np.float64)

    # Identity
    rotations.append(np.eye(3))

    # 8 rotations about the 4 body diagonals (3-fold symmetry axes), by +/-120 deg
    body_diags = [
        np.array([1, 1, 1]),
        np.array([1, -1, -1]),
        np.array([-1, 1, -1]),
        np.array([-1, -1, 1]),
    ]
    for d in body_diags:
        rotations.append(rot_axis_angle(d, 2 * np.pi / 3))
        rotations.append(rot_axis_angle(d, 4 * np.pi / 3))

    # 3 rotations by 180 deg about the coordinate axes
    for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
        rotations.append(rot_axis_angle(axis, np.pi))

    assert len(rotations) == 12
    return np.stack(rotations, axis=0)


class TetrahedralBackend(RotationOnlyBackend):
    """Tetrahedral group T ≅ A_4 (12 proper rotations)."""

    def _enumerate_rotations(self) -> np.ndarray:
        return _enumerate_tetrahedral()

    def _get_generators(self) -> List[int]:
        # Use first non-identity elements as generators
        return [1, 9]  # 3-fold and 2-fold


# ---------------------------------------------------------------------------
# Octahedral group O (S_4, 24 elements)
# ---------------------------------------------------------------------------

def _enumerate_octahedral() -> np.ndarray:
    """Enumerate the 24 proper rotations of the octahedral group.

    The octahedral group consists of all 3x3 signed permutation matrices with
    determinant +1.  There are 3! * 2^3 = 48 signed permutation matrices in
    total; exactly half (24) have det = +1.
    """
    rotations = []

    for perm in permutations([0, 1, 2]):
        for s0 in (1, -1):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    R = np.zeros((3, 3), dtype=np.float64)
                    for col_idx, (row_idx, s) in enumerate(zip(perm, [s0, s1, s2])):
                        R[row_idx, col_idx] = float(s)
                    if np.linalg.det(R) > 0.5:
                        rotations.append(R)

    assert len(rotations) == 24, f"Expected 24, got {len(rotations)}"
    return np.stack(rotations, axis=0)


def _find_rotation_generator(rotations: np.ndarray, target_R: np.ndarray, tol: float = 1e-5) -> int:
    """Find the index of a rotation matching target_R in the list."""
    for i, R in enumerate(rotations):
        if np.abs(R - target_R).max() < tol:
            return i
    raise ValueError("Generator not found in rotation list")


class OctahedralBackend(RotationOnlyBackend):
    """Octahedral group O ≅ S_4 (24 proper rotations)."""

    def _enumerate_rotations(self) -> np.ndarray:
        return _enumerate_octahedral()

    def _get_generators(self) -> List[int]:
        # Use 90° rotations about z and x axes as generators.
        # These are order-4 elements that together generate the full octahedral group.
        R_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        R_x90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        rots = self._rotations
        try:
            iz = _find_rotation_generator(rots, R_z90)
            ix = _find_rotation_generator(rots, R_x90)
            return [iz, ix]
        except ValueError:
            # Fall back to first two non-identity elements
            return [1, 2]

    def subgroup(self, name: str) -> RotationOnlyBackend:
        if name == "T":
            return TetrahedralBackend()
        if name.startswith("Z"):
            m = int(name[1:])
            return CyclicBackend(m)
        if name == "D4":
            return DihedralBackend(4)
        raise ValueError(f"Unknown subgroup name '{name}'")


# ---------------------------------------------------------------------------
# Dihedral group D_n (2n elements in SO(3): n rotations + n pi-rotations)
# ---------------------------------------------------------------------------

class DihedralBackend(RotationOnlyBackend):
    """Dihedral group D_n viewed as a subgroup of SO(3).

    Elements: n rotations about z-axis + n pi-rotations about horizontal axes.
    """

    def __init__(self, n: int):
        self.n = n
        super().__init__()

    def _enumerate_rotations(self) -> np.ndarray:
        rotations = []
        for k in range(self.n):
            theta = 2 * np.pi * k / self.n
            c, s = np.cos(theta), np.sin(theta)
            # z-axis rotation
            rotations.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64))
            # pi-rotation about the axis (cos(theta*k/2), sin(theta*k/2), 0)
            phi = np.pi * k / self.n
            cx, sx = np.cos(phi), np.sin(phi)
            axis = np.array([cx, sx, 0.0])
            # Rodrigues formula for pi-rotation
            R = 2 * np.outer(axis, axis) - np.eye(3)
            rotations.append(R.astype(np.float64))
        return np.stack(rotations, axis=0)

    def _get_generators(self) -> List[int]:
        return [1, 2]


# ---------------------------------------------------------------------------
# Icosahedral group I (A_5, 60 elements)
# ---------------------------------------------------------------------------

def _enumerate_icosahedral() -> np.ndarray:
    """Enumerate the 60 proper rotations of the icosahedral group.

    Generated by BFS from identity using 5-fold and 3-fold generators.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(float(angle))
        x, y, z = axis
        return np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
        ], dtype=np.float64)

    # 5-fold axis: the z-axis (rotation by 2*pi/5)
    gen1 = rot_axis_angle(np.array([0., 0., 1.]), 2 * np.pi / 5)
    # 3-fold axis: a vertex of the icosahedron
    v = np.array([0., 1., phi]) / np.linalg.norm([0., 1., phi])
    gen2 = rot_axis_angle(v, 2 * np.pi / 3)

    generators = [gen1, gen2]

    def mat_close(A, B, tol=1e-6):
        return np.abs(A - B).max() < tol

    def find_in_list(rots, R):
        for i, r in enumerate(rots):
            if mat_close(r, R):
                return i
        return -1

    rotations = [np.eye(3)]
    queue = deque([0])
    while queue:
        idx = queue.popleft()
        for gen in generators:
            for R_new in [gen @ rotations[idx], gen.T @ rotations[idx]]:
                if find_in_list(rotations, R_new) == -1:
                    rotations.append(R_new)
                    queue.append(len(rotations) - 1)

    assert len(rotations) == 60, f"Expected 60 icosahedral rotations, got {len(rotations)}"
    return np.stack(rotations, axis=0)


class IcosahedralBackend(RotationOnlyBackend):
    """Icosahedral group I ≅ A_5 (60 proper rotations)."""

    def _enumerate_rotations(self) -> np.ndarray:
        return _enumerate_icosahedral()

    def _get_generators(self) -> List[int]:
        return [1, 2]

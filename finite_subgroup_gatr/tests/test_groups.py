"""Tests for finite group backends.

Checks:
  - multiplication table associativity
  - inverse correctness
  - identity correctness
  - neighborhood contains identity
  - subgroup closure
  - coset partition correctness
"""

import pytest
import torch
import numpy as np

from finite_subgroup_gatr.backends.rotation_backend import (
    CyclicBackend,
    OctahedralBackend,
    TetrahedralBackend,
)


@pytest.fixture(params=["cyclic4", "cyclic8", "tetrahedral", "octahedral"])
def backend(request):
    name = request.param
    if name == "cyclic4":
        return CyclicBackend(4)
    if name == "cyclic8":
        return CyclicBackend(8)
    if name == "tetrahedral":
        return TetrahedralBackend()
    if name == "octahedral":
        return OctahedralBackend()
    raise ValueError(name)


class TestMultiplicationTable:
    def test_identity_is_neutral(self, backend):
        """e * g = g * e = g for all g."""
        e = backend.identity()
        for g in range(backend.size()):
            assert backend.multiply(e, g) == g, f"e*g != g at g={g}"
            assert backend.multiply(g, e) == g, f"g*e != g at g={g}"

    def test_associativity_sampled(self, backend):
        """(a*b)*c = a*(b*c) for random triples."""
        rng = np.random.default_rng(42)
        G = backend.size()
        for _ in range(min(100, G ** 2)):
            a, b, c = rng.integers(0, G, size=3).tolist()
            ab = backend.multiply(a, b)
            ab_c = backend.multiply(ab, c)
            bc = backend.multiply(b, c)
            a_bc = backend.multiply(a, bc)
            assert ab_c == a_bc, f"Associativity failed: ({a}*{b})*{c}={ab_c} != {a}*({b}*{c})={a_bc}"

    def test_inverse(self, backend):
        """g * g^{-1} = e and g^{-1} * g = e."""
        e = backend.identity()
        for g in range(backend.size()):
            gi = backend.inverse(g)
            assert backend.multiply(g, gi) == e, f"g*g_inv != e at g={g}"
            assert backend.multiply(gi, g) == e, f"g_inv*g != e at g={g}"

    def test_group_size(self, backend):
        """Group size is positive and consistent with elements list."""
        assert backend.size() > 0
        assert len(backend.elements()) == backend.size()


class TestNeighborhood:
    def test_contains_identity(self, backend):
        """Identity should be in every neighborhood."""
        for r in range(1, 4):
            nbr = backend.neighborhood(r)
            assert backend.identity() in nbr.tolist(), f"Identity not in N_{r}"

    def test_monotone(self, backend):
        """N_{r} ⊆ N_{r+1}."""
        nbr1 = set(backend.neighborhood(1).tolist())
        nbr2 = set(backend.neighborhood(2).tolist())
        assert nbr1.issubset(nbr2), "Neighborhood is not monotone"

    def test_full_group(self, backend):
        """For large enough radius, neighborhood = full group."""
        nbr = backend.neighborhood(backend.size())
        assert len(nbr) == backend.size(), "Large neighborhood should be full group"


class TestSubgroupCoset:
    def test_cyclic_subgroup_closure(self):
        """Z_4 as subgroup of Z_8 should be closed."""
        big = CyclicBackend(8)
        sub = big.subgroup("Z4")
        # All products of subgroup elements should remain in subgroup
        H_indices = list(range(sub.size()))
        for h1 in H_indices:
            for h2 in H_indices:
                h1h2 = sub.multiply(h1, h2)
                assert h1h2 in H_indices, f"Subgroup not closed: {h1}*{h2}={h1h2} not in H"

    def test_coset_partition_covers_group(self):
        """Cosets should partition the full group."""
        big = CyclicBackend(8)
        sub = big.subgroup("Z4")
        cosets = big.coset_partition(sub)
        all_indices = set()
        for coset in cosets:
            for idx in coset:
                assert idx not in all_indices, "Coset element appears in multiple cosets"
                all_indices.add(idx)
        assert all_indices == set(range(big.size())), "Cosets do not cover the full group"

    def test_coset_sizes(self):
        """All cosets should have the same size |H|."""
        big = CyclicBackend(8)
        sub = big.subgroup("Z4")
        cosets = big.coset_partition(sub)
        sizes = [len(c) for c in cosets]
        assert len(set(sizes)) == 1, f"Cosets have different sizes: {sizes}"
        assert sizes[0] == sub.size()


class TestMultivectorAction:
    def test_action_is_rotation(self, backend):
        """Action on 3D vectors should preserve norms."""
        pts = torch.randn(10, 3)
        for g in range(min(backend.size(), 5)):
            rotated = backend.action_on_points(g, pts)
            orig_norms = pts.norm(dim=-1)
            rot_norms = rotated.norm(dim=-1)
            assert torch.allclose(orig_norms, rot_norms, atol=1e-5), \
                f"Rotation should preserve norms at g={g}"

    def test_mv_action_consistency(self, backend):
        """action_on_points then embed vs embed then action_on_multivectors should match."""
        from finite_subgroup_gatr.pga.embed import embed_points
        from gatr.interface.point import extract_point
        pts = torch.randn(5, 3)
        mv = embed_points(pts)

        for g in range(min(backend.size(), 3)):
            # Method 1: transform point then embed
            pts_rot = backend.action_on_points(g, pts)
            mv_from_pts = embed_points(pts_rot)

            # Method 2: embed then apply MV action matrix
            mv_from_mv = backend.action_on_multivectors(g, mv)

            # Compare trivector parts (grade-3 carries point info)
            assert torch.allclose(mv_from_pts[..., 11:15], mv_from_mv[..., 11:15], atol=1e-4), \
                f"Point embedding action mismatch at g={g}"

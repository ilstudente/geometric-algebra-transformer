"""Tests for PGA actions and embeddings."""

import pytest
import torch

from finite_subgroup_gatr.backends.rotation_backend import OctahedralBackend, CyclicBackend
from finite_subgroup_gatr.pga.embed import embed_points, embed_vectors, embed_planes
from gatr.interface.point import extract_point


@pytest.fixture
def octahedral():
    return OctahedralBackend()


@pytest.fixture
def cyclic4():
    return CyclicBackend(4)


class TestPointEmbedding:
    def test_embed_extract_roundtrip(self):
        """embed_points then extract_point should recover original coordinates."""
        pts = torch.randn(8, 3)
        mv = embed_points(pts)
        pts_back = extract_point(mv)
        assert torch.allclose(pts, pts_back, atol=1e-5)

    def test_embed_shape(self):
        pts = torch.randn(4, 3, 3)  # batch shape [4, 3]
        mv = embed_points(pts)
        assert mv.shape == (4, 3, 16)


class TestGroupActionConsistency:
    """Verify action_on_points ∘ embed matches embed ∘ action_on_multivectors."""

    def test_octahedral_point_action(self, octahedral):
        pts = torch.randn(10, 3)
        mv = embed_points(pts)

        for g in range(octahedral.size()):
            pts_rot = octahedral.action_on_points(g, pts)
            mv_rot_from_pts = embed_points(pts_rot)

            mv_rot_from_mv = octahedral.action_on_multivectors(g, mv)

            # Grade-3 components should match (points embed as grade-3 trivectors)
            assert torch.allclose(
                mv_rot_from_pts[..., 11:15],
                mv_rot_from_mv[..., 11:15],
                atol=1e-4,
            ), f"Point embedding/action mismatch at g={g}"

    def test_cyclic4_vector_action(self, cyclic4):
        """Vectors should transform correctly under cyclic rotations."""
        vecs = torch.randn(5, 3)
        mv = embed_vectors(vecs)

        for g in range(cyclic4.size()):
            vecs_rot = cyclic4.action_on_points(g, vecs)  # same rotation for free vectors
            mv_rot_from_pts = embed_vectors(vecs_rot)
            mv_rot_from_mv = cyclic4.action_on_multivectors(g, mv)

            # Grade-1 spatial components should match
            assert torch.allclose(
                mv_rot_from_pts[..., 2:5],  # e1, e2, e3
                mv_rot_from_mv[..., 2:5],
                atol=1e-4,
            ), f"Vector action mismatch at g={g}"


class TestGroupHomomorphism:
    """action(g) ∘ action(h) = action(g*h)."""

    def test_octahedral_composition(self, octahedral):
        mv = torch.randn(5, 16)
        G = octahedral.size()

        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(20):
            g, h = rng.integers(0, G, size=2).tolist()
            gh = octahedral.multiply(g, h)

            # Apply h then g
            mv_h = octahedral.action_on_multivectors(h, mv)
            mv_gh_seq = octahedral.action_on_multivectors(g, mv_h)

            # Apply g*h directly
            mv_gh = octahedral.action_on_multivectors(gh, mv)

            assert torch.allclose(mv_gh_seq, mv_gh, atol=1e-4), \
                f"Composition mismatch at g={g}, h={h}, gh={gh}"

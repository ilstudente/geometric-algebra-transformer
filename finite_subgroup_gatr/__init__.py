"""FiniteSubgroupGATr: Finite-subgroup rebuild of GATr with GM-CNN token mixing.

This package replaces GATr's exact continuous E(3)-equivariance with finite-group
approximate equivariance while preserving PGA multivector hidden states, geometric
product, equivariant join, and gated nonlinearities.

Three variants are provided:
  - Variant A: GM-only token mixer (no attention)
  - Variant B: GM + discrete multivector attention
  - Variant C: Axial split (GM over symmetry axis, attention over other axes)
"""

from finite_subgroup_gatr.backends.rotation_backend import (
    CyclicBackend,
    OctahedralBackend,
    TetrahedralBackend,
    IcosahedralBackend,
)
from finite_subgroup_gatr.backends.grid_rotation_backend import GridRotationBackend
from finite_subgroup_gatr.models.variant_a_gm_only import VariantA
from finite_subgroup_gatr.models.variant_b_gm_attention import VariantB
from finite_subgroup_gatr.models.variant_c_axial import VariantC

__all__ = [
    "CyclicBackend",
    "OctahedralBackend",
    "TetrahedralBackend",
    "IcosahedralBackend",
    "GridRotationBackend",
    "VariantA",
    "VariantB",
    "VariantC",
]

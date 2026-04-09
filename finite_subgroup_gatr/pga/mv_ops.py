"""PGA multivector operations.

Thin wrappers around GATr primitives with shape-generic, batch-safe interfaces.
All functions operate on the last dimension (size 16) of the input tensor.
"""

import torch
from torch import Tensor

# Re-export from GATr primitives
from gatr.primitives.bilinear import geometric_product  # noqa: F401
from gatr.primitives.dual import equivariant_join as _equivariant_join
from gatr.primitives.invariants import inner_product as _inner_product
from gatr.primitives.normalization import equi_layer_norm as _equi_layer_norm
from gatr.primitives.nonlinearities import gated_gelu


# PGA basis grade structure
# Index 0         : scalar  (grade 0)
# Indices 1-4     : vectors (grade 1) — e0, e1, e2, e3
# Indices 5-10    : bivectors (grade 2) — e01, e02, e03, e12, e13, e23
# Indices 11-14   : trivectors (grade 3) — e012, e013, e023, e123
# Index 15        : pseudoscalar (grade 4)

_GRADE_SLICES = [
    slice(0, 1),    # grade 0
    slice(1, 5),    # grade 1
    slice(5, 11),   # grade 2
    slice(11, 15),  # grade 3
    slice(15, 16),  # grade 4
]


def grade_project(mv: Tensor, grade: int) -> Tensor:
    """Project multivector to a single grade.

    Parameters
    ----------
    mv : Tensor [..., 16]
    grade : int in {0, 1, 2, 3, 4}

    Returns
    -------
    Tensor [..., 16] with only the components of the given grade non-zero.
    """
    out = torch.zeros_like(mv)
    out[..., _GRADE_SLICES[grade]] = mv[..., _GRADE_SLICES[grade]]
    return out


def equi_join(x: Tensor, y: Tensor, ref: Tensor) -> Tensor:
    """Equivariant join of two PGA multivectors.

    Parameters
    ----------
    x, y : Tensor [..., 16]
    ref  : Tensor [..., 16]  reference multivector

    Returns
    -------
    Tensor [..., 16]
    """
    return _equivariant_join(x, y, ref)


def mv_inner_product(x: Tensor, y: Tensor) -> Tensor:
    """GA inner product <x, y>.

    Parameters
    ----------
    x, y : Tensor [..., 16]

    Returns
    -------
    Tensor [..., 1]
    """
    return _inner_product(x, y).unsqueeze(-1)


def mv_layer_norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Equivariant layer normalization for multivectors.

    Normalizes over the channel dimension (second-to-last).

    Parameters
    ----------
    x : Tensor [..., C, 16]
    eps : float

    Returns
    -------
    Tensor [..., C, 16]
    """
    return _equi_layer_norm(x, channel_dim=-2, epsilon=eps)


def gated_gelu_mv(x: Tensor, gates: Tensor) -> Tensor:
    """Equivariant gated GELU.

    Parameters
    ----------
    x     : Tensor [..., 16]  multivector to gate
    gates : Tensor [..., 1]   scalar gate values

    Returns
    -------
    Tensor [..., 16]
    """
    return gated_gelu(x, gates)

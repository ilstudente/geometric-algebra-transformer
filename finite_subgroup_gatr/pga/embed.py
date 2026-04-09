"""PGA embedding functions.

Maps geometric primitives (points, vectors, planes, translations, rotations)
into 16-component PGA multivectors following GATr conventions.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor


class PGAEmbedder(nn.Module):
    """Learnable linear embedding from geometric primitives to multivector channels.

    Given raw geometric data (points, vectors, etc.) embedded into PGA multivectors,
    this module projects them into C_mv multivector channels and C_s scalar channels.

    Parameters
    ----------
    in_mv_channels : int
        Number of input PGA multivectors per token.
    out_mv_channels : int
        Number of output multivector channels.
    in_s_channels : int or None
        Number of input scalar features. None means no scalars.
    out_s_channels : int or None
        Number of output scalar channels.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        in_s_channels: int | None = None,
        out_s_channels: int | None = None,
    ):
        super().__init__()
        from gatr.layers.linear import EquiLinear
        self.linear = EquiLinear(
            in_mv_channels,
            out_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
        )

    def forward(
        self,
        mv: Tensor,
        scalars: Tensor | None = None,
    ):
        """
        Parameters
        ----------
        mv : Tensor [..., in_mv_channels, 16]
        scalars : Tensor [..., in_s_channels] or None

        Returns
        -------
        mv_out : Tensor [..., out_mv_channels, 16]
        s_out  : Tensor [..., out_s_channels] or None
        """
        return self.linear(mv, scalars)


# ---------------------------------------------------------------------------
# Static embedding helpers
# ---------------------------------------------------------------------------

def embed_points(pts: Tensor) -> Tensor:
    """Embed 3D points as PGA trivectors.

    Parameters
    ----------
    pts : Tensor [..., 3]

    Returns
    -------
    Tensor [..., 16]
    """
    from gatr.interface.point import embed_point
    return embed_point(pts)


def embed_vectors(vecs: Tensor) -> Tensor:
    """Embed 3D free vectors as PGA vectors (grade 1, spatial part).

    Parameters
    ----------
    vecs : Tensor [..., 3]

    Returns
    -------
    Tensor [..., 16]
    """
    batch = vecs.shape[:-1]
    mv = torch.zeros(*batch, 16, dtype=vecs.dtype, device=vecs.device)
    # Spatial basis vectors: e1=index2, e2=index3, e3=index4
    mv[..., 2] = vecs[..., 0]
    mv[..., 3] = vecs[..., 1]
    mv[..., 4] = vecs[..., 2]
    return mv


def embed_planes(normals: Tensor, offsets: Tensor) -> Tensor:
    """Embed planes (n, d) as PGA bivectors n·x = d.

    A plane with unit normal n = (nx, ny, nz) and offset d is
    embedded as: nx*e1 + ny*e2 + nz*e3 + d*e0.

    Parameters
    ----------
    normals : Tensor [..., 3]
    offsets : Tensor [..., 1]

    Returns
    -------
    Tensor [..., 16]
    """
    batch = normals.shape[:-1]
    mv = torch.zeros(*batch, 16, dtype=normals.dtype, device=normals.device)
    # e0 = index 1, e1 = index 2, e2 = index 3, e3 = index 4
    mv[..., 1] = offsets[..., 0]
    mv[..., 2] = normals[..., 0]
    mv[..., 3] = normals[..., 1]
    mv[..., 4] = normals[..., 2]
    return mv


def embed_translations(t: Tensor) -> Tensor:
    """Embed 3D translation vectors as PGA motors.

    T = 1 - (t/2) e_{0i}  for each spatial component.

    Parameters
    ----------
    t : Tensor [..., 3]

    Returns
    -------
    Tensor [..., 16]
    """
    batch = t.shape[:-1]
    mv = torch.zeros(*batch, 16, dtype=t.dtype, device=t.device)
    mv[..., 0] = 1.0           # scalar = 1
    mv[..., 5] = -t[..., 0] / 2  # e01
    mv[..., 6] = -t[..., 1] / 2  # e02
    mv[..., 7] = -t[..., 2] / 2  # e03
    return mv


def embed_rotations(quaternion: Tensor) -> Tensor:
    """Embed rotation quaternion as PGA rotor.

    Parameters
    ----------
    quaternion : Tensor [..., 4]  (i, j, k, w) Hamilton convention

    Returns
    -------
    Tensor [..., 16]
    """
    from gatr.interface.rotation import embed_rotation
    return embed_rotation(quaternion)


def embed_scalars(s: Tensor) -> Tensor:
    """Embed scalars into the grade-0 component of a multivector.

    Parameters
    ----------
    s : Tensor [..., 1]

    Returns
    -------
    Tensor [..., 16]
    """
    batch = s.shape[:-1]
    mv = torch.zeros(*batch, 16, dtype=s.dtype, device=s.device)
    mv[..., 0] = s[..., 0]
    return mv

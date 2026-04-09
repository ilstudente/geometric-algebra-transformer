"""Discrete multivector attention for finite-group token sequences.

Attention logits use the PGA inner product between Q and K multivectors,
plus optional scalar logits, following GATr's geometric attention design.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from gatr.layers.linear import EquiLinear
from gatr.primitives.invariants import inner_product
from gatr.primitives.normalization import equi_layer_norm


class DiscreteMVAttention(nn.Module):
    """Multi-head attention over PGA multivector token sequences.

    Attention logits:
        a_ij = sum_heads sum_c <q_ic, k_jc>  + (scalar_logit_ij if enabled)

    Value aggregation is equivariant: v' = softmax(a) @ v.

    Parameters
    ----------
    c_mv : int
        Input multivector channels.
    c_s : int or None
        Input scalar channels.
    num_heads : int
    head_dim_mv : int
        Multivector channels per head (for Q/K/V).
    use_scalar_logits : bool
        Whether to add scalar inner products to the attention logits.
    use_distance_bias : bool
        Whether to add a learned relative-position bias based on the group structure.
    """

    def __init__(
        self,
        c_mv: int,
        c_s: int | None,
        num_heads: int = 4,
        head_dim_mv: int | None = None,
        use_scalar_logits: bool = True,
        use_distance_bias: bool = False,
        scalar_attn_bundle: "nn.Module | None" = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.c_mv = c_mv
        self.c_s = c_s
        self.use_scalar_logits = use_scalar_logits
        self.use_distance_bias = use_distance_bias

        if head_dim_mv is None:
            head_dim_mv = max(1, c_mv // num_heads)
        self.head_dim_mv = head_dim_mv

        total_qk_mv = num_heads * head_dim_mv

        # Scalar layer norm
        if c_s is not None:
            self.s_norm = nn.LayerNorm(c_s)
        else:
            self.s_norm = None

        # Q, K, V projections
        self.q_proj = EquiLinear(c_mv, total_qk_mv, in_s_channels=c_s, out_s_channels=None)
        self.k_proj = EquiLinear(c_mv, total_qk_mv, in_s_channels=c_s, out_s_channels=None)
        self.v_proj = EquiLinear(c_mv, c_mv, in_s_channels=c_s, out_s_channels=None)

        # Scalar Q/K for scalar logits — use injected bundle or build dense defaults
        self._scalar_attn_bundle = scalar_attn_bundle
        if use_scalar_logits and c_s is not None:
            head_dim_s = max(1, (c_s or 0) // num_heads)
            self.head_dim_s = head_dim_s
            total_qk_s = num_heads * head_dim_s
            if scalar_attn_bundle is None:
                self.q_s_proj: nn.Module | None = nn.Linear(c_s, total_qk_s)
                self.k_s_proj: nn.Module | None = nn.Linear(c_s, total_qk_s)
            else:
                # Projections are handled by the bundle; keep None as sentinels
                self.q_s_proj = None
                self.k_s_proj = None
        else:
            self.head_dim_s = 0
            self.q_s_proj = None
            self.k_s_proj = None

        # Output projection
        self.out_proj = EquiLinear(c_mv, c_mv, in_s_channels=c_s, out_s_channels=c_s)

        # Scale factor
        self.scale = (head_dim_mv + self.head_dim_s) ** -0.5

    def forward(
        self,
        x_mv: Tensor,
        x_s: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        x_mv : Tensor [B, X, C_mv, 16]
        x_s  : Tensor [B, X, C_s] or None
        mask : Tensor [B, X, X] or None  (additive mask, -inf for masked positions)

        Returns
        -------
        out_mv : Tensor [B, X, C_mv, 16]
        out_s  : Tensor [B, X, C_s] or None
        """
        B, X, C_mv, _ = x_mv.shape

        # Layer norms
        h_mv = equi_layer_norm(x_mv)
        h_s = self.s_norm(x_s) if (x_s is not None and self.s_norm is not None) else x_s

        # Q, K, V
        Q_mv, _ = self.q_proj(h_mv, scalars=h_s)  # [B, X, H*head_dim_mv, 16]
        K_mv, _ = self.k_proj(h_mv, scalars=h_s)
        V_mv, _ = self.v_proj(h_mv, scalars=h_s)  # [B, X, C_mv, 16]

        H = self.num_heads
        hd = self.head_dim_mv

        # Reshape to [B, X, H, head_dim_mv, 16]
        Q_mv = Q_mv.reshape(B, X, H, hd, 16)
        K_mv = K_mv.reshape(B, X, H, hd, 16)

        # Attention logits from MV inner product: sum over channels and heads
        # <Q_i, K_j> for each head -> [B, H, X, X]
        # inner_product returns [..., 1], so we squeeze
        # Q: [B, X, H, hd, 16] -> [B, H, X, hd, 16]
        Q_mv_t = Q_mv.permute(0, 2, 1, 3, 4)  # [B, H, X, hd, 16]
        K_mv_t = K_mv.permute(0, 2, 1, 3, 4)  # [B, H, X, hd, 16]

        # inner_product: [..., 16] x [..., 16] -> [...]
        # We need [B, H, X_i, X_j] from [B, H, X, hd, 16]
        # Expand: Q -> [B, H, X, 1, hd, 16], K -> [B, H, 1, X, hd, 16]
        Q_exp = Q_mv_t.unsqueeze(3)  # [B, H, X, 1, hd, 16]
        K_exp = K_mv_t.unsqueeze(2)  # [B, H, 1, X, hd, 16]
        logits_mv = inner_product(Q_exp, K_exp).sum(dim=-2)  # [B, H, X, X, 1] -> sum over hd -> [B, H, X, X]
        logits_mv = logits_mv.squeeze(-1)

        logits = logits_mv  # [B, H, X, X]

        # Add scalar logits — via injected bundle or default dense projections
        if self.use_scalar_logits and h_s is not None:
            if self._scalar_attn_bundle is not None:
                Q_s_raw = self._scalar_attn_bundle.forward_q(h_s)  # [B, X, H*hd_s]
                K_s_raw = self._scalar_attn_bundle.forward_k(h_s)
            elif self.q_s_proj is not None and self.k_s_proj is not None:
                Q_s_raw = self.q_s_proj(h_s)
                K_s_raw = self.k_s_proj(h_s)
            else:
                Q_s_raw = K_s_raw = None

            if Q_s_raw is not None and K_s_raw is not None:
                Q_s = Q_s_raw.reshape(B, X, H, self.head_dim_s)  # [B, X, H, hd_s]
                K_s = K_s_raw.reshape(B, X, H, self.head_dim_s)
                Q_s_t = Q_s.permute(0, 2, 1, 3)  # [B, H, X, hd_s]
                K_s_t = K_s.permute(0, 2, 1, 3)
                logits_s = torch.einsum("bhid, bhjd -> bhij", Q_s_t, K_s_t)  # [B, H, X, X]
                logits = logits + logits_s

        logits = logits * self.scale

        # Apply mask
        if mask is not None:
            # mask: [B, X, X] or [B, 1, X, X]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            logits = logits + mask

        # Softmax
        attn = torch.softmax(logits, dim=-1)  # [B, H, X, X]

        # Value aggregation: aggregate over heads then reshape
        # V: [B, X, C_mv, 16] -> we average over heads (or sum head contributions)
        # Simpler: use average attention over heads
        attn_avg = attn.mean(dim=1)  # [B, X, X]

        # Weighted sum of values
        # V: [B, X, C_mv, 16], attn_avg: [B, X_out, X_in]
        # out[b, i, c, :] = sum_j attn[b, i, j] * V[b, j, c, :]
        V_exp = V_mv.unsqueeze(1).expand(B, X, X, C_mv, 16)  # [B, X_out, X_in, C_mv, 16]
        attn_exp = attn_avg.unsqueeze(-1).unsqueeze(-1)  # [B, X_out, X_in, 1, 1]
        out_mv = (attn_exp * V_exp).sum(dim=2)  # [B, X, C_mv, 16]

        # Output projection
        out_mv, out_s = self.out_proj(out_mv, scalars=h_s)

        return out_mv, out_s

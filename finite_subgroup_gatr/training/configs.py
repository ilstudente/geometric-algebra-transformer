"""Configuration dataclasses for FiniteSubgroupGATr experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BackboneConfig:
    """Configuration for the backbone architecture.

    Attributes
    ----------
    variant : {"A", "B", "C"}
    backend : {"cyclic_n", "octahedral", "tetrahedral", "icosahedral", "grid_rotation"}
    backend_n : int
        Order for cyclic group or grid size.
    c_mv : int
        Hidden multivector channels.
    c_s : int or None
        Hidden scalar channels.
    num_blocks : int
        Number of backbone blocks (or cycles for variant C).
    neighborhood_radius : int
    num_heads : int
        Number of attention heads (used in variants B and C).
    error_mode : {"none", "low_rank", "dense_penalized"}
    error_rank : int
    use_scalar_logits : bool
    """

    variant: str = "A"
    backend: str = "octahedral"
    backend_n: int = 8
    c_mv: int = 8
    c_s: int = 32
    num_blocks: int = 6
    neighborhood_radius: int = 1
    num_heads: int = 4
    error_mode: str = "none"
    error_rank: int = 4
    use_scalar_logits: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Attributes
    ----------
    lr : float
    weight_decay : float
    batch_size : int
    max_epochs : int
    lambda_eq : float
    lambda_error : float
    lambda_disp : float
    eq_num_samples : int
    gradient_clip : float or None
    """

    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    lambda_eq: float = 0.0
    lambda_error: float = 0.0
    lambda_disp: float = 0.0
    eq_num_samples: int = 8
    gradient_clip: Optional[float] = 1.0
    seed: int = 42


@dataclass
class ScalarPathGMConfig:
    """Configuration for Variant 3: scalar-path-only GM replacement.

    Controls which scalar linear maps are replaced with GM-structured operators,
    and which approximate-equivariance mode they use.

    Attributes
    ----------
    enabled : bool
        Master switch.  If False, all scalar maps remain dense (no-op).
    scope : {"mlp_only", "attn_only", "all_scalar", "scalar_interface"}
        Which scalar maps to replace:
          "mlp_only"        – scalar FFN in GeoMLP only (Scope A).
          "attn_only"       – scalar Q/K projections in attention only (Scope B).
          "all_scalar"      – all scalar linear maps (Scope C).
          "scalar_interface"– additionally replaces scalar↔MV interface maps (Scope D).
    mode : {"dense", "gm_exact", "gm_approx"}
        "dense"     – standard linear layers (useful to verify the scope switch works).
        "gm_exact"  – exact GM structure, no residual.
        "gm_approx" – GM + residual parameterised by error_mode / error_rank.
    neighborhood_radius : int
    error_mode : {"none", "low_rank", "dense_penalized"}
        Residual structure for approximate mode.
    error_rank : int
        Rank for low_rank mode.
    lambda_error : float
        Coefficient for the structural residual penalty in the training loss.
    lambda_eq : float
        Coefficient for the scalar behavioural equivariance loss.
    token_axis_name : str
        Human-readable name of the symmetry axis (used in logging only).
    """

    enabled: bool = True
    scope: str = "mlp_only"
    mode: str = "gm_exact"
    neighborhood_radius: int = 1
    error_mode: str = "none"
    error_rank: int = 4
    lambda_error: float = 0.0
    lambda_eq: float = 0.0
    token_axis_name: str = "symmetry"


@dataclass
class AblationConfig:
    """Configuration for the ablation matrix.

    Each flag toggles a specific component on/off.
    """

    # GM structure
    exact_gm: bool = True
    low_rank_residual: bool = False
    dense_penalized_residual: bool = False

    # Loss terms
    use_equivariance_loss: bool = False

    # Architecture components
    use_geo_mlp: bool = True
    use_scalar_path: bool = True
    grade_aware_mixing: bool = True

    # Backend
    backend: str = "octahedral"  # or "cyclic_8" etc.

    def to_backbone_config_kwargs(self) -> dict:
        """Convert to BackboneConfig keyword arguments."""
        if self.low_rank_residual:
            error_mode = "low_rank"
        elif self.dense_penalized_residual:
            error_mode = "dense_penalized"
        else:
            error_mode = "none"
        return {
            "error_mode": error_mode,
            "backend": self.backend,
        }

    def to_training_config_kwargs(self) -> dict:
        """Convert to TrainingConfig keyword arguments."""
        return {
            "lambda_eq": 0.01 if self.use_equivariance_loss else 0.0,
        }

#!/usr/bin/env python3
"""Entry script for the n-body experiment with FiniteSubgroupGATr.

Usage (from repo root)
----------------------
Activate environment first::

    conda activate gatr  # or: source activate /path/to/gatr/env

Basic run (Octahedral group, VariantA, exact equivariance)::

    python scripts/finite_subgroup_nbody.py \\
        base_dir=/path/to/base \\
        run_name=oct_varA_exact \\
        seed=42

Override to VariantB (GM + attention)::

    python scripts/finite_subgroup_nbody.py \\
        base_dir=/path/to/base \\
        run_name=oct_varB \\
        seed=42 \\
        model=finite_subgroup_variant_b_nbody

Use a Cyclic group (Z_8) instead::

    python scripts/finite_subgroup_nbody.py \\
        base_dir=/path/to/base \\
        run_name=cyclic8_varA \\
        seed=42 \\
        model.backend._target_=finite_subgroup_gatr.backends.rotation_backend.CyclicBackend \\
        model.backend.n=8 \\
        model.net.backend._target_=finite_subgroup_gatr.backends.rotation_backend.CyclicBackend \\
        model.net.backend.n=8

Enable approximate equivariance (low-rank perturbation)::

    python scripts/finite_subgroup_nbody.py \\
        base_dir=/path/to/base \\
        run_name=oct_varA_approx \\
        seed=42 \\
        model.net.error_mode=low_rank \\
        model.net.error_rank=8

Data directory must contain ``train.npz``, ``val.npz``, ``test.npz``.
Generate them with ``python scripts/generate_nbody_dataset.py``.
"""

import sys
from pathlib import Path

# Ensure the repo root is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig

from finite_subgroup_gatr.experiments.nbody_experiment import NBodyFiniteSubgroupExperiment


@hydra.main(
    config_path="../config",
    config_name="finite_subgroup_nbody",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    exp = NBodyFiniteSubgroupExperiment(cfg)
    exp()


if __name__ == "__main__":
    main()

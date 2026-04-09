#!/bin/bash
#SBATCH --job-name=fs_gatr_variants
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/n/netscratch/mweber_lab/Everyone/dettel/logs/variants_%j.log

# Experiment 1: Variant A (GM-only) vs Variant B (GM + attention)
# Fixed: OctahedralBackend, exact equivariance, c_mv=16, num_blocks=4, nbhd_radius=1

set -euo pipefail

REPO=/n/home10/dettel/geometric-algebra-transformer
BASEDIR=/n/netscratch/mweber_lab/Everyone/dettel/fs_gatr_experiments
PYTHON=/n/home10/dettel/conda_envs/gatr/bin/python
SEED=42

mkdir -p /n/netscratch/mweber_lab/Everyone/dettel/logs
cd "$REPO"

module load cuda/12.9.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
export LD_LIBRARY_PATH=${CUDNN_PATH:-}/lib:$LD_LIBRARY_PATH

echo "=== Variant Comparison Experiment ==="
echo "Node: $SLURMD_NODENAME  |  Job: $SLURM_JOB_ID  |  $(date)"
$PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# -----------------------------------------------------------------
# Run 1: Variant A — GM-only, exact equivariance
# -----------------------------------------------------------------
echo ""
echo "--- Run 1/2: VariantA (GM-only), Octahedral, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_octahedral_exact" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody
echo "Run 1 complete: $(date)"

# -----------------------------------------------------------------
# Run 2: Variant B — GM + attention, exact equivariance
# -----------------------------------------------------------------
echo ""
echo "--- Run 2/2: VariantB (GM+attention), Octahedral, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varB_octahedral_exact" \
    seed=${SEED} \
    model=finite_subgroup_variant_b_nbody
echo "Run 2 complete: $(date)"

echo ""
echo "=== All variant runs complete: $(date) ==="

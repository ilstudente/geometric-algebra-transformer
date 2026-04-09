#!/bin/bash
#SBATCH --job-name=fs_gatr_equiv
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/n/netscratch/mweber_lab/Everyone/dettel/logs/equiv_ablation_%j.log

# Experiment 3: Equivariance ablation
# Fixed: VariantA, OctahedralBackend, c_mv=16, num_blocks=4, nbhd_radius=1
# Vary:  exact | low_rank r=4 | low_rank r=8 | low_rank r=16 | dense_penalized

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

echo "=== Equivariance Ablation Experiment ==="
echo "Node: $SLURMD_NODENAME  |  Job: $SLURM_JOB_ID  |  $(date)"
$PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# -----------------------------------------------------------------
# Run 1: Exact equivariance (baseline)
# -----------------------------------------------------------------
echo ""
echo "--- Run 1/5: exact equivariance ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_octahedral_exact" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=none
echo "Run 1 complete: $(date)"

# -----------------------------------------------------------------
# Run 2: Low-rank perturbation, rank 4
# -----------------------------------------------------------------
echo ""
echo "--- Run 2/5: low_rank r=4 ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_octahedral_lowrank4" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=low_rank \
    model.net.error_rank=4
echo "Run 2 complete: $(date)"

# -----------------------------------------------------------------
# Run 3: Low-rank perturbation, rank 8
# -----------------------------------------------------------------
echo ""
echo "--- Run 3/5: low_rank r=8 ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_octahedral_lowrank8" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=low_rank \
    model.net.error_rank=8
echo "Run 3 complete: $(date)"

# -----------------------------------------------------------------
# Run 4: Low-rank perturbation, rank 16
# -----------------------------------------------------------------
echo ""
echo "--- Run 4/5: low_rank r=16 ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_octahedral_lowrank16" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=low_rank \
    model.net.error_rank=16
echo "Run 4 complete: $(date)"

# -----------------------------------------------------------------
# Run 5: Dense penalized (full non-equivariant perturbation)
# -----------------------------------------------------------------
echo ""
echo "--- Run 5/5: dense_penalized ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_octahedral_dense" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=dense_penalized
echo "Run 5 complete: $(date)"

echo ""
echo "=== All equivariance ablation runs complete: $(date) ==="

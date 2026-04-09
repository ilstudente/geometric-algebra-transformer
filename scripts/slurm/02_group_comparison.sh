#!/bin/bash
#SBATCH --job-name=fs_gatr_groups
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/n/netscratch/mweber_lab/Everyone/dettel/logs/groups_%j.log

# Experiment 2: Group comparison
# Fixed: VariantA, exact equivariance, c_mv=16, num_blocks=4, nbhd_radius=1
# Vary:  Cyclic Z_8 (|G|=8), Tetrahedral (|G|=12), Octahedral (|G|=24), Icosahedral (|G|=60)

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

echo "=== Group Comparison Experiment ==="
echo "Node: $SLURMD_NODENAME  |  Job: $SLURM_JOB_ID  |  $(date)"
$PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# -----------------------------------------------------------------
# Run 1: Cyclic Z_8 (order 8)
# -----------------------------------------------------------------
echo ""
echo "--- Run 1/4: Cyclic Z_8 (|G|=8), VariantA, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_cyclic8_exact" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    'model.backend._target_=finite_subgroup_gatr.backends.rotation_backend.CyclicBackend' \
    ++model.backend.n=8 \
    'model.net.backend._target_=finite_subgroup_gatr.backends.rotation_backend.CyclicBackend' \
    ++model.net.backend.n=8
echo "Run 1 complete: $(date)"

# -----------------------------------------------------------------
# Run 2: Tetrahedral (order 12)
# -----------------------------------------------------------------
echo ""
echo "--- Run 2/4: Tetrahedral (|G|=12), VariantA, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_tetrahedral_exact" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    'model.backend._target_=finite_subgroup_gatr.backends.rotation_backend.TetrahedralBackend' \
    'model.net.backend._target_=finite_subgroup_gatr.backends.rotation_backend.TetrahedralBackend'
echo "Run 2 complete: $(date)"

# -----------------------------------------------------------------
# Run 3: Octahedral (order 24) — default, included for fair comparison
# -----------------------------------------------------------------
echo ""
echo "--- Run 3/4: Octahedral (|G|=24), VariantA, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_octahedral_exact" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody
echo "Run 3 complete: $(date)"

# -----------------------------------------------------------------
# Run 4: Icosahedral (order 60)
# -----------------------------------------------------------------
echo ""
echo "--- Run 4/4: Icosahedral (|G|=60), VariantA, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="varA_icosahedral_exact" \
    seed=${SEED} \
    model=finite_subgroup_variant_a_nbody \
    'model.backend._target_=finite_subgroup_gatr.backends.rotation_backend.IcosahedralBackend' \
    'model.net.backend._target_=finite_subgroup_gatr.backends.rotation_backend.IcosahedralBackend'
echo "Run 4 complete: $(date)"

echo ""
echo "=== All group runs complete: $(date) ==="

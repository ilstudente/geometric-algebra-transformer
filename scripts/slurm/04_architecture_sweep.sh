#!/bin/bash
#SBATCH --job-name=fs_gatr_arch
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=/n/netscratch/mweber_lab/Everyone/dettel/logs/arch_sweep_%j.log

# Experiment 4: Architecture sweep
# Fixed: VariantA, OctahedralBackend, exact equivariance
# Sweep: c_mv in {8, 16, 32}  x  num_blocks in {2, 4, 8}  (nbhd_radius fixed at 1)
#    then: nbhd_radius in {1, 2, 3} with c_mv=16, num_blocks=4

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

echo "=== Architecture Sweep Experiment ==="
echo "Node: $SLURMD_NODENAME  |  Job: $SLURM_JOB_ID  |  $(date)"
$PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# -----------------------------------------------------------------
# Part A: vary c_mv with num_blocks=4, nbhd_radius=1
# -----------------------------------------------------------------

for C_MV in 8 16 32; do
    echo ""
    echo "--- c_mv=${C_MV}, num_blocks=4, nbhd_radius=1 ---"
    $PYTHON scripts/finite_subgroup_nbody.py \
        base_dir="${BASEDIR}" \
        run_name="arch_cmv${C_MV}_blocks4_r1" \
        seed=${SEED} \
        model=finite_subgroup_variant_a_nbody \
        model.net.c_mv=${C_MV} \
        model.net.num_blocks=4 \
        model.net.neighborhood_radius=1
    echo "Done c_mv=${C_MV}: $(date)"
done

# -----------------------------------------------------------------
# Part B: vary num_blocks with c_mv=16, nbhd_radius=1
# (skip blocks=4 since it was already run above as arch_cmv16_blocks4_r1)
# -----------------------------------------------------------------

for BLOCKS in 2 8; do
    echo ""
    echo "--- c_mv=16, num_blocks=${BLOCKS}, nbhd_radius=1 ---"
    $PYTHON scripts/finite_subgroup_nbody.py \
        base_dir="${BASEDIR}" \
        run_name="arch_cmv16_blocks${BLOCKS}_r1" \
        seed=${SEED} \
        model=finite_subgroup_variant_a_nbody \
        model.net.c_mv=16 \
        model.net.num_blocks=${BLOCKS} \
        model.net.neighborhood_radius=1
    echo "Done blocks=${BLOCKS}: $(date)"
done

# -----------------------------------------------------------------
# Part C: vary neighborhood_radius with c_mv=16, num_blocks=4
# (skip radius=1, already covered above)
# -----------------------------------------------------------------

for RADIUS in 2 3; do
    echo ""
    echo "--- c_mv=16, num_blocks=4, nbhd_radius=${RADIUS} ---"
    $PYTHON scripts/finite_subgroup_nbody.py \
        base_dir="${BASEDIR}" \
        run_name="arch_cmv16_blocks4_r${RADIUS}" \
        seed=${SEED} \
        model=finite_subgroup_variant_a_nbody \
        model.net.c_mv=16 \
        model.net.num_blocks=4 \
        model.net.neighborhood_radius=${RADIUS}
    echo "Done radius=${RADIUS}: $(date)"
done

echo ""
echo "=== All architecture sweep runs complete: $(date) ==="

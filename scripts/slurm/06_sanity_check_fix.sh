#!/bin/bash
#SBATCH --job-name=fs_sanity
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/n/netscratch/mweber_lab/Everyone/dettel/logs/sanity_%j.log

# Sanity check for the equivariant-unlifting fix.
# Runs 5 small-scale experiments (2000 steps, batchsize 64) to verify that
# all group/variant combinations now produce decreasing validation loss instead
# of the trivial zeros-predictor MSE (~406 for octahedral, ~272 for cyclic).
#
# Expected: all runs should reach val MSE well below 400 within 2000 steps.

set -euo pipefail

REPO=/n/home10/dettel/geometric-algebra-transformer
BASEDIR=/n/netscratch/mweber_lab/Everyone/dettel/fs_gatr_experiments
PYTHON=/n/home10/dettel/conda_envs/gatr/bin/python
SEED=42
STEPS=2000

mkdir -p /n/netscratch/mweber_lab/Everyone/dettel/logs
cd "$REPO"

module load cuda/12.9.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
export LD_LIBRARY_PATH=${CUDNN_PATH:-}/lib:$LD_LIBRARY_PATH

echo "=== Sanity Check: Equivariant Unlifting Fix ==="
echo "Node: $SLURMD_NODENAME  |  Job: $SLURM_JOB_ID  |  $(date)"
$PYTHON -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo "Steps per run: $STEPS  (trivial-predictor MSE: oct~406, cyclic~272)"

# -----------------------------------------------------------------
# Run 1: VariantA + Octahedral + exact  (was stuck at 406.21)
# -----------------------------------------------------------------
echo ""
echo "--- Run 1/5: VariantA, Octahedral, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="sanity_varA_oct_exact" \
    seed=${SEED} \
    training.steps=${STEPS} \
    training.early_stopping=false \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=none
echo "Run 1 complete: $(date)"

# -----------------------------------------------------------------
# Run 2: VariantA + Octahedral + low_rank  (was stuck at 406.21)
# -----------------------------------------------------------------
echo ""
echo "--- Run 2/5: VariantA, Octahedral, low_rank r=4 ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="sanity_varA_oct_lowrank4" \
    seed=${SEED} \
    training.steps=${STEPS} \
    training.early_stopping=false \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=low_rank \
    model.net.error_rank=4
echo "Run 2 complete: $(date)"

# -----------------------------------------------------------------
# Run 3: VariantA + Octahedral + dense_penalized  (was stuck at 406.21)
# -----------------------------------------------------------------
echo ""
echo "--- Run 3/5: VariantA, Octahedral, dense_penalized ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="sanity_varA_oct_dense" \
    seed=${SEED} \
    training.steps=${STEPS} \
    training.early_stopping=false \
    model=finite_subgroup_variant_a_nbody \
    model.net.error_mode=dense_penalized
echo "Run 3 complete: $(date)"

# -----------------------------------------------------------------
# Run 4: VariantA + Cyclic Z_8 + exact  (was stuck at 271.83)
# -----------------------------------------------------------------
echo ""
echo "--- Run 4/5: VariantA, Cyclic Z_8, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="sanity_varA_cyclic8_exact" \
    seed=${SEED} \
    training.steps=${STEPS} \
    training.early_stopping=false \
    model=finite_subgroup_variant_a_nbody \
    'model.backend._target_=finite_subgroup_gatr.backends.rotation_backend.CyclicBackend' \
    ++model.backend.n=8 \
    'model.net.backend._target_=finite_subgroup_gatr.backends.rotation_backend.CyclicBackend' \
    ++model.net.backend.n=8
echo "Run 4 complete: $(date)"

# -----------------------------------------------------------------
# Run 5: VariantB + Octahedral + exact  (new variant, also needs check)
# -----------------------------------------------------------------
echo ""
echo "--- Run 5/5: VariantB, Octahedral, exact ---"
$PYTHON scripts/finite_subgroup_nbody.py \
    base_dir="${BASEDIR}" \
    run_name="sanity_varB_oct_exact" \
    seed=${SEED} \
    training.steps=${STEPS} \
    training.early_stopping=false \
    model=finite_subgroup_variant_b_nbody
echo "Run 5 complete: $(date)"

echo ""
echo "=== All sanity runs complete: $(date) ==="
echo ""
echo "--- Results summary ---"
for run in sanity_varA_oct_exact sanity_varA_oct_lowrank4 sanity_varA_oct_dense sanity_varA_cyclic8_exact sanity_varB_oct_exact; do
    csv="${BASEDIR}/experiments/finite_subgroup_nbody/${run}/metrics/eval_eval.csv"
    if [ -f "$csv" ]; then
        mse=$(tail -1 "$csv" | cut -d',' -f2)
        echo "  ${run}: eval MSE = ${mse}"
    else
        echo "  ${run}: no CSV found"
    fi
done

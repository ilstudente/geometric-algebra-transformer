#!/bin/bash
#SBATCH --job-name=fs_gatr_data
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/n/netscratch/mweber_lab/Everyone/dettel/logs/generate_dataset_%j.log

set -euo pipefail

REPO=/n/home10/dettel/geometric-algebra-transformer
BASEDIR=/n/netscratch/mweber_lab/Everyone/dettel/fs_gatr_experiments
PYTHON=/n/home10/dettel/conda_envs/gatr/bin/python

mkdir -p "${BASEDIR}/data/nbody"
mkdir -p /n/netscratch/mweber_lab/Everyone/dettel/logs

cd "$REPO"

if [ -f "${BASEDIR}/data/nbody/train.npz" ]; then
    echo "Dataset already exists at ${BASEDIR}/data/nbody/ — skipping generation."
else
    echo "Generating n-body dataset at ${BASEDIR}/data/nbody/ ..."
    $PYTHON scripts/generate_nbody_dataset.py \
        base_dir="${BASEDIR}" \
        seed=42
    echo "Dataset generation complete."
fi

ls -lh "${BASEDIR}/data/nbody/"

#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=c2f-surface
#SBATCH --output=surface-%j.out
#SBATCH --error=surface-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${C2F_PROJECT:?}" "${C2F_PYTHON:?}"
: "${C2F_CACHE:?}" "${C2F_DATA_DIR:?}" "${C2F_WARM_START:?}"
: "${C2F_SURFACE_RESULTS:?}" "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$C2F_PYTHON" submission/surface_comparison.py --cache "$C2F_CACHE" \
  --dataset "$C2F_DATA_DIR" --warm-start "$C2F_WARM_START" --output "$C2F_SURFACE_RESULTS"

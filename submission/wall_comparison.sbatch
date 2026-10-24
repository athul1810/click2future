#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=c2f-wall-pair
#SBATCH --output=wall-pair-%j.out
#SBATCH --error=wall-pair-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set the committed source directory}"
: "${C2F_PYTHON:?Set the validated TC2 interpreter}"
: "${C2F_DATA_DIR:?Set the frozen training corpus}"
: "${C2F_WALL_RESULTS:?Set a fresh comparison directory}"
: "${C2F_SOURCE_REVISION:?Set the committed source revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$C2F_PYTHON" submission/wall_comparison.py \
  --dataset "$C2F_DATA_DIR" --output "$C2F_WALL_RESULTS"

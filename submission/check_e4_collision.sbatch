#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=c2f-e4-collision
#SBATCH --output=e4-collision-%j.out
#SBATCH --error=e4-collision-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?}"
: "${C2F_PYTHON:?}"
: "${C2F_PILOT_ROOT:?}"
: "${C2F_E4_CHECK:?}"
: "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$C2F_PYTHON" submission/check_e4_collision.py --root "$C2F_PILOT_ROOT" --output "$C2F_E4_CHECK"

#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=c2f-learn
#SBATCH --output=tc2-learn-%j.out
#SBATCH --error=tc2-learn-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
: "${C2F_PROJECT:?Set the source release}"
: "${C2F_PILOT_ROOT:?Set a new experiment directory}"
: "${C2F_SOURCE_REVISION:?Set the committed source revision}"
cd -- "$C2F_PROJECT"
export UV_CACHE_DIR="$HOME/.cache/click2future/uv"
export UV_PROJECT_ENVIRONMENT="$C2F_PROJECT/.venv"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$HOME/.conda/envs/c2f-bootstrap/bin/uv" sync --locked --extra cu128 --no-dev \
    --python "$HOME/.conda/envs/c2f-bootstrap/bin/python"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv/bin/python -m click2future.training.tc2_pilot "$1" --root "$C2F_PILOT_ROOT"

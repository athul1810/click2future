#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=c2f-curve
#SBATCH --output=tc2-curve-%j.out
#SBATCH --error=tc2-curve-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
: "${C2F_PROJECT:?Set source release}"
: "${C2F_DATASET:?Set training dataset}"
: "${C2F_RUN_DIR:?Set a new run directory}"
: "${C2F_BASELINE_CHECKPOINT:?Set the completed 2000-step diagnostic checkpoint}"
: "${C2F_SOURCE_REVISION:?Set source revision}"
cd -- "$C2F_PROJECT"
export UV_CACHE_DIR="$HOME/.cache/click2future/uv"
export UV_PROJECT_ENVIRONMENT="$C2F_PROJECT/.venv"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$HOME/.conda/envs/c2f-bootstrap/bin/uv" sync --locked --extra cu128 --no-dev \
    --python "$HOME/.conda/envs/c2f-bootstrap/bin/python"
.venv/bin/python -m click2future.training.diagnose_sampling --require-slurm --device cuda \
    --checkpoint "$C2F_BASELINE_CHECKPOINT" --dataset "$C2F_DATASET" \
    --output-dir "$C2F_RUN_DIR/baseline-sampler" --sampling-steps 32 128 1000
# Independent restart at a longer declared horizon; never relabel as exact resume.
.venv/bin/python -m click2future.training.train_diffusion --require-slurm --device cuda \
    --dataset "$C2F_DATASET" --run-dir "$C2F_RUN_DIR" \
    --config configs/training/learning_curve.json
.venv/bin/python -m click2future.training.diagnose_sampling --require-slurm --device cuda \
    --checkpoint "$C2F_RUN_DIR/checkpoint.pt" --dataset "$C2F_DATASET" \
    --output-dir "$C2F_RUN_DIR/final-sampler" --sampling-steps 32 128

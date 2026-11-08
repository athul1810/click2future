#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
: "${C2F_PROJECT:?}" "${C2F_PYTHON:?}" "${C2F_CACHE:?}"
: "${C2F_DATA_DIR:?}" "${C2F_RESULTS:?}" "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
args=()
[[ "${C2F_RESUME:-0}" != 1 ]] || args+=(--resume)
[[ "${C2F_EVAL_ONLY:-0}" != 1 ]] || args+=(--evaluate-only)
exec "$C2F_PYTHON" -m click2future.training.train_contact_dynamics \
  --cache "$C2F_CACHE" --dataset "$C2F_DATA_DIR" --output "$C2F_RESULTS" "${args[@]}"

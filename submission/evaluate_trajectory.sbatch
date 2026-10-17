#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=c2f-trajectory-eval
#SBATCH --output=trajectory-eval-%j.out
#SBATCH --error=trajectory-eval-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set the committed source directory}"
: "${C2F_PYTHON:?Set the validated TC2 interpreter}"
: "${C2F_DATA_DIR:?Set the frozen training corpus}"
: "${C2F_TRAJECTORY_CHECKPOINT:?Set the selected trained checkpoint}"
: "${C2F_TRAJECTORY_RESULTS:?Set a new validation output directory}"
: "${C2F_SOURCE_REVISION:?Set the evaluation source revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$C2F_PYTHON" -m click2future.evaluation.trajectory_pilot \
  --dataset "$C2F_DATA_DIR" --checkpoint "$C2F_TRAJECTORY_CHECKPOINT" \
  --output "$C2F_TRAJECTORY_RESULTS" --device cuda --steps 16 "$@"

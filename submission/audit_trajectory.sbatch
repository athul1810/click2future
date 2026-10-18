#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --job-name=c2f-geometry-audit
#SBATCH --output=geometry-audit-%j.out
#SBATCH --error=geometry-audit-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set the committed source directory}"
: "${C2F_PYTHON:?Set the validated TC2 interpreter}"
: "${C2F_DATA_DIR:?Set the frozen training corpus}"
: "${C2F_TRAJECTORY_CHECKPOINT:?Set the exact evaluated E5 checkpoint}"
: "${C2F_AUDIT_PLAN:?Set the original frozen validation plan}"
: "${C2F_AUDIT_PREDICTIONS:?Set the original predictions manifest}"
: "${C2F_AUDIT_OUTPUT:?Set a new audit output directory}"
: "${C2F_SOURCE_REVISION:?Set the audit source revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$C2F_PYTHON" -m click2future.evaluation.trajectory_audit \
  --plan "$C2F_AUDIT_PLAN" --predictions "$C2F_AUDIT_PREDICTIONS" \
  --dataset "$C2F_DATA_DIR" --checkpoint "$C2F_TRAJECTORY_CHECKPOINT" \
  --output "$C2F_AUDIT_OUTPUT" --device cuda

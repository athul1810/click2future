#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=c2f-contact
#SBATCH --output=contact-%j.out
#SBATCH --error=contact-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set the committed source directory}"
: "${C2F_PYTHON:?Set the validated TC2 interpreter}"
: "${C2F_DATA_DIR:?Set the frozen training corpus}"
: "${C2F_RUN_DIR:?Set a new contact run directory}"
: "${C2F_SOURCE_REVISION:?Set the training source revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$C2F_PYTHON" -m click2future.training.train_contact \
  --dataset "$C2F_DATA_DIR" --run-dir "$C2F_RUN_DIR" \
  --config configs/training/contact_pilot.json "$@"

#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=c2f-pilot-eval
#SBATCH --output=pilot-eval-%j.out
#SBATCH --error=pilot-eval-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set the committed source directory}"
: "${C2F_PYTHON:?Set the validated TC2 interpreter}"
: "${C2F_PILOT_ROOT:?Set the completed pilot directory}"
: "${C2F_PILOT_RESULTS:?Set a new evaluation directory}"
: "${C2F_SOURCE_REVISION:?Set the evaluation source revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" -m click2future.evaluation.pilot \
  --root "$C2F_PILOT_ROOT" --output "$C2F_PILOT_RESULTS" --device cuda

#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=c2f-sampler-study
#SBATCH --output=sampler-study-%j.out
#SBATCH --error=sampler-study-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?}"
: "${C2F_PYTHON:?}"
: "${C2F_PILOT_ROOT:?}"
: "${C2F_PILOT_RESULTS:?}"
: "${C2F_STUDY_RESULTS:?}"
: "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" submission/sampler_study.py --root "$C2F_PILOT_ROOT" \
  --parent-plan "$C2F_PILOT_RESULTS/plan.json" --output "$C2F_STUDY_RESULTS" --device cuda

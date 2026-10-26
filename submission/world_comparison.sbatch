#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:55:00
#SBATCH --job-name=c2f-world
#SBATCH --output=world-%j.out
#SBATCH --error=world-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set immutable source}"
: "${C2F_PYTHON:?Set validated interpreter}"
: "${C2F_DATA_DIR:?Set frozen original corpus}"
: "${C2F_SUPPLEMENT:?Set retained training action supplement}"
: "${C2F_WARM_START:?Set frozen EMA checkpoint}"
: "${C2F_WORLD_RESULTS:?Set fresh comparison output}"
: "${C2F_SOURCE_REVISION:?Set committed revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$C2F_PYTHON" submission/world_comparison.py \
  --dataset "$C2F_DATA_DIR" --supplement "$C2F_SUPPLEMENT" \
  --warm-start "$C2F_WARM_START" --output "$C2F_WORLD_RESULTS"

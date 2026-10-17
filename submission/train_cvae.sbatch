#!/bin/bash
#SBATCH --job-name=c2f-cvae
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=tc2-cvae-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${C2F_DATA_DIR:?Set C2F_DATA_DIR to the generated dataset folder}"
: "${C2F_RUN_DIR:?Set C2F_RUN_DIR to a new run folder}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec .venv/bin/python -m click2future.training.train_cvae \
  --data-dir "$C2F_DATA_DIR" --output-dir "$C2F_RUN_DIR" "$@"

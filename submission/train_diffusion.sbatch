#!/bin/bash
#SBATCH --job-name=c2f-diffusion
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=tc2-diffusion-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${C2F_DATA_DIR:?Set the generated dataset folder}"
: "${C2F_RUN_DIR:?Set the run folder}"
: "${C2F_TRAIN_CONFIG:?Set the immutable training JSON}"
: "${C2F_SOURCE_REVISION:?Set the source commit or source archive hash}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec .venv/bin/python - "$@" <<'PY'
import os
import sys

import torch

from click2future.training.train_diffusion import main

if os.environ.get("SLURM_JOB_PARTITION") != "MGPU-TC2" or torch.cuda.device_count() != 1:
    raise SystemExit("Training requires one visible GPU in a TC2 MGPU-TC2 allocation")
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
main([
    "--dataset", os.environ["C2F_DATA_DIR"],
    "--run-dir", os.environ["C2F_RUN_DIR"],
    "--config", os.environ["C2F_TRAIN_CONFIG"],
    *sys.argv[1:], "--device", "cuda", "--require-slurm",
])
PY

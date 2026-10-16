#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=c2f-target
#SBATCH --output=tc2-target-%j.out
#SBATCH --error=tc2-target-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
: "${C2F_PROJECT:?Set committed source release}"
: "${C2F_PYTHON:?Set the already validated interpreter}"
: "${C2F_DATASET:?Set training dataset}"
: "${C2F_RUN_DIR:?Set a new run directory}"
: "${C2F_SOURCE_REVISION:?Set committed revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Reuse pinned dependencies without changing another release's editable install.
"$C2F_PYTHON" -c 'import os,pathlib,click2future.training.train_diffusion as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"])/"src"); print(m.__file__)'
"$C2F_PYTHON" -m click2future.training.train_diffusion --require-slurm --device cuda \
    --dataset "$C2F_DATASET" --run-dir "$C2F_RUN_DIR" \
    --config configs/training/velocity_probe.json
"$C2F_PYTHON" -m click2future.training.diagnose_sampling --require-slurm --device cuda \
    --checkpoint "$C2F_RUN_DIR/checkpoint.pt" --dataset "$C2F_DATASET" \
    --output-dir "$C2F_RUN_DIR/sampler" --sampling-steps 32 128

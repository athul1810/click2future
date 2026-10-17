#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=c2f-repr
#SBATCH --output=tc2-repr-%j.out
#SBATCH --error=tc2-repr-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
: "${C2F_PROJECT:?Set committed source release}"
: "${C2F_PYTHON:?Set validated interpreter}"
: "${C2F_DATASET:?Set frozen one-clip diagnostic corpus}"
: "${C2F_RUN_DIR:?Set a new run directory}"
: "${C2F_SOURCE_REVISION:?Set committed revision}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" -c 'import os,pathlib,click2future.training.train_diffusion as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"])/"src"); print(m.__file__)'
if [[ ! -e "$C2F_DATASET" ]]; then
    "$C2F_PYTHON" -m click2future.cli generate-dataset --output-dir "$C2F_DATASET" \
        --group-count 10 --variants-per-group 2 --seed 226000 \
        --config configs/simulator/one_clip_diagnostic.toml
fi
"$C2F_PYTHON" -m click2future.training.compare_representation --device cuda \
    --dataset "$C2F_DATASET" --output-dir "$C2F_RUN_DIR" \
    --config configs/training/diffusion_one_clip_boundary.json

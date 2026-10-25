#!/bin/bash -l
# Validation-only rehearsal of the final E0/E2/E3 run: one shared plan, every run sampled
# and scored together, then both action horizons. Test splits are never opened.
# Usage: sbatch cluster/final_rehearsal.sbatch, with C2F_RUNS naming a sample_suite run spec.
# C2F_OUT must not exist. LPIPS needs the AlexNet weights cached in ~/.cache/torch/hub/checkpoints.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=c2f-final-rehearsal
#SBATCH --output=rehearsal-%j.out
#SBATCH --error=rehearsal-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?}" "${C2F_PYTHON:?}" "${C2F_DATA:?}" "${C2F_RUNS:?}" "${C2F_OUT:?}" "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" -c 'import os,pathlib,click2future as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"]).resolve()/"src")'
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$C2F_SOURCE_REVISION"
exec "$C2F_PYTHON" cluster/final_rehearsal.py --project "$C2F_PROJECT" --data "$C2F_DATA" \
    --runs "$C2F_RUNS" --output "$C2F_OUT" --device cuda

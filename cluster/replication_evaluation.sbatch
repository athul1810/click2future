#!/bin/bash -l
# D017 held-out replication: generate (or re-check) the separate corpus, freeze every plan,
# then sample and score the frozen runs. C2F_RUNS (E0/E2/E3 spec) and/or C2F_POSE_RUNS (E10)
# name the checkpoints. C2F_OUT must not exist. LPIPS needs the cached AlexNet weights.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --job-name=c2f-replication
#SBATCH --output=replication-%j.out
#SBATCH --error=replication-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?}" "${C2F_PYTHON:?}" "${C2F_DATA:?}" "${C2F_HELD_OUT:?}" "${C2F_OUT:?}" "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" -c 'import os,pathlib,click2future as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"]).resolve()/"src")'
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$C2F_SOURCE_REVISION"
runs=()
[ -n "${C2F_RUNS:-}" ] && runs+=(--runs "$C2F_RUNS")
[ -n "${C2F_POSE_RUNS:-}" ] && runs+=(--pose-runs "$C2F_POSE_RUNS")
exec "$C2F_PYTHON" cluster/replication_evaluation.py --project "$C2F_PROJECT" --training-data "$C2F_DATA" \
    --held-out "$C2F_HELD_OUT" --output "$C2F_OUT" --device cuda "${runs[@]}"

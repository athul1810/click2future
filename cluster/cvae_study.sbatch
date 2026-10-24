#!/bin/bash -l
# Train and evaluate one or more CVAE configs in sequence on the frozen dataset-0.3 pilot.
# Usage: sbatch cluster/cvae_study.sbatch <config.yaml> [<config.yaml> ...]
# Each config gets $C2F_STUDY_DIR/<config-name>/{train,eval,force-probe.json}; existing runs are skipped.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=c2f-cvae-study
#SBATCH --output=cvae-study-%j.out
#SBATCH --error=cvae-study-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set the committed source directory}"
: "${C2F_PYTHON:?Set the validated TC2 interpreter}"
: "${C2F_PILOT_ROOT:?Set the pilot directory (prepared here if absent)}"
: "${C2F_STUDY_DIR:?Set the study output directory}"
: "${C2F_SOURCE_REVISION:?Set the committed source revision}"
(( $# )) || { echo 'Pass at least one CVAE config' >&2; exit 2; }
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" -c 'import os,pathlib,click2future.training.train_cvae as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"]).resolve()/"src")'
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$C2F_SOURCE_REVISION"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if [[ ! -f "$C2F_PILOT_ROOT/pilot.json" ]]; then
    "$C2F_PYTHON" -m click2future.training.multiscene_pilot --root "$C2F_PILOT_ROOT" --prepare-only
fi
mkdir -p "$C2F_STUDY_DIR"
status=0
for config in "$@"; do
    name=$(basename -- "$config")
    run="$C2F_STUDY_DIR/${name%.*}"
    if [[ -e "$run" ]]; then
        printf 'Skipping %s: %s already exists\n' "$config" "$run"
        continue
    fi
    printf '\n=== %s: train %s ===\n' "$(date -u +%H:%M:%S)" "$config"
    if ! "$C2F_PYTHON" -m click2future.training.train_cvae \
        --data-dir "$C2F_PILOT_ROOT/data" --output-dir "$run/train" --config "$config"; then
        status=1
        continue
    fi
    printf '\n=== %s: evaluate %s ===\n' "$(date -u +%H:%M:%S)" "$config"
    "$C2F_PYTHON" submission/cvae_evaluation.py --root "$C2F_PILOT_ROOT" \
        --checkpoint "$run/train/best.pt" --output "$run/eval" --device cuda || status=1
    printf '\n=== %s: force probe %s ===\n' "$(date -u +%H:%M:%S)" "$config"
    "$C2F_PYTHON" submission/cvae_force_probe.py --dataset "$C2F_PILOT_ROOT/data" \
        --checkpoint "$run/train/best.pt" --output "$run/force-probe.json" --device cuda || status=1
done
printf '\n=== %s: done, status %s ===\n' "$(date -u +%H:%M:%S)" "$status"
exit "$status"

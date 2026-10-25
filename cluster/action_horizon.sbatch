#!/bin/bash -l
# Score finished CVAE study runs at the 0.5-second window and the full clip with
# click2future.evaluation.action_horizon, using each run's original saved plans and videos.
# Usage: sbatch cluster/action_horizon.sbatch <run-name> [<run-name> ...]
# Writes $C2F_STUDY_DIR/<run>/action-horizon-{validation,selection}.json; existing reports are kept.
# The scorer runs on CPU; the GPU request follows the partition's required job flags.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=c2f-action-horizon
#SBATCH --output=action-horizon-%j.out
#SBATCH --error=action-horizon-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?Set the committed source directory}"
: "${C2F_PYTHON:?Set the validated TC2 interpreter}"
: "${C2F_STUDY_DIR:?Set the study output directory}"
(( $# )) || { echo 'Pass at least one run name' >&2; exit 2; }
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$C2F_PYTHON" -c 'import os,pathlib,click2future.evaluation.action_horizon as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"]).resolve()/"src")'
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${C2F_SOURCE_REVISION:-unrecorded}"
status=0
for run in "$@"; do
    eval_dir="$C2F_STUDY_DIR/$run/eval"
    for suite in validation selection; do
        if [[ $suite == validation ]]; then
            plan="$eval_dir/plan.json" samples="$eval_dir/validation-samples"
        else
            plan="$eval_dir/target-controls/plan.json" samples="$eval_dir/selected-object-samples"
        fi
        output="$C2F_STUDY_DIR/$run/action-horizon-$suite.json"
        if [[ -e "$output" ]]; then
            printf 'Skipping %s: %s exists\n' "$run $suite" "$output"
            continue
        fi
        printf '\n=== %s: %s %s ===\n' "$(date -u +%H:%M:%S)" "$run" "$suite"
        "$C2F_PYTHON" -m click2future.evaluation.action_horizon \
            --plan "$plan" --predictions "$samples/predictions.json" --output "$output" || status=1
    done
done
printf '\n=== %s: done, status %s ===\n' "$(date -u +%H:%M:%S)" "$status"
exit "$status"

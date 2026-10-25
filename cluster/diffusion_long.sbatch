#!/bin/bash -l
# Post-test follow-up, validation only: train E2/E3 seed 0 for 40,000 updates, the CVAE's
# 40,000-clip training budget, with everything else as in the final configs. Checkpoints are
# kept at 10k, 20k and 40k updates (the cosine schedule runs over 40k, so the first two are
# mid-schedule) and each is scored on validation with cluster/final_rehearsal.py.
# Usage: sbatch cluster/diffusion_long.sbatch. Finished stages and evaluations are skipped.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=c2f-diffusion-long
#SBATCH --output=diffusion-long-%j.out
#SBATCH --error=diffusion-long-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?}" "${C2F_PYTHON:?}" "${C2F_DATA:?}" "${C2F_STUDY_DIR:?}" "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" -c 'import os,pathlib,click2future as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"]).resolve()/"src")'
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$C2F_SOURCE_REVISION"
milestones=(10000 20000 40000)
for conditioning in numeric spatial; do
    method=E2
    [[ $conditioning == spatial ]] && method=E3
    run="$C2F_STUDY_DIR/$method-seed0"
    for until in "${milestones[@]}"; do
        [[ -e "$run/step-$until.pt" ]] && continue
        resume=()
        [[ -e "$run/checkpoint.pt" ]] && resume=(--resume "$run/checkpoint.pt")
        printf '\n=== %s: train %s to %s ===\n' "$(date -u +%H:%M:%S)" "$method" "$until"
        "$C2F_PYTHON" -m click2future.training.train_diffusion --dataset "$C2F_DATA" \
            --run-dir "$run" --config "configs/training/diffusion_long/${conditioning}_seed0.json" \
            --device cuda --require-slurm --until "$until" ${resume[@]+"${resume[@]}"}
        cp -- "$run/checkpoint.pt" "$run/step-$until.pt"
    done
done
for until in "${milestones[@]}"; do
    output="$C2F_STUDY_DIR/validation-$until"
    [[ -e "$output" ]] && continue
    spec="$C2F_STUDY_DIR/runs-$until.json"
    printf '{"sampling_steps": 32, "runs": [%s, %s]}\n' \
        "{\"method\": \"E2\", \"run_id\": \"E2-seed0-$until\", \"checkpoint\": \"$C2F_STUDY_DIR/E2-seed0/step-$until.pt\"}" \
        "{\"method\": \"E3\", \"run_id\": \"E3-seed0-$until\", \"checkpoint\": \"$C2F_STUDY_DIR/E3-seed0/step-$until.pt\"}" > "$spec"
    printf '\n=== %s: validation at %s ===\n' "$(date -u +%H:%M:%S)" "$until"
    "$C2F_PYTHON" cluster/final_rehearsal.py --project "$C2F_PROJECT" --data "$C2F_DATA" \
        --runs "$spec" --output "$output" --device cuda
done

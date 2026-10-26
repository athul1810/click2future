#!/bin/bash -l
# Train E2/E3 for 40,000 updates, matching the CVAE's 40,000 training clips (not its updates or
# GPU time), with everything else as in the final configs (D015). Checkpoints are kept at 10k,
# 20k and 40k updates (the cosine schedule runs over 40k, so the first two are mid-schedule) and
# each is scored on validation with cluster/final_rehearsal.py. C2F_SEEDS lists the training
# seeds, default 0.
# Usage: sbatch cluster/diffusion_long.sbatch. Finished stages and evaluations are skipped.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
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
for seed in ${C2F_SEEDS:-0}; do
    for conditioning in numeric spatial; do
        method=E2
        [[ $conditioning == spatial ]] && method=E3
        run="$C2F_STUDY_DIR/$method-seed$seed"
        for until in "${milestones[@]}"; do
            [[ -e "$run/step-$until.pt" ]] && continue
            resume=()
            [[ -e "$run/checkpoint.pt" ]] && resume=(--resume "$run/checkpoint.pt")
            printf '\n=== %s: train %s seed %s to %s ===\n' "$(date -u +%H:%M:%S)" "$method" "$seed" "$until"
            "$C2F_PYTHON" -m click2future.training.train_diffusion --dataset "$C2F_DATA" \
                --run-dir "$run" --config "configs/training/diffusion_long/${conditioning}_seed$seed.json" \
                --device cuda --require-slurm --until "$until" ${resume[@]+"${resume[@]}"}
            cp -- "$run/checkpoint.pt" "$run/step-$until.pt"
        done
    done
    for until in "${milestones[@]}"; do
        output="$C2F_STUDY_DIR/validation-seed$seed-$until"
        [[ -e "$output" ]] && continue
        spec="$C2F_STUDY_DIR/runs-seed$seed-$until.json"
        printf '{"sampling_steps": 32, "runs": [%s, %s]}\n' \
            "{\"method\": \"E2\", \"run_id\": \"E2-seed$seed-$until\", \"checkpoint\": \"$C2F_STUDY_DIR/E2-seed$seed/step-$until.pt\"}" \
            "{\"method\": \"E3\", \"run_id\": \"E3-seed$seed-$until\", \"checkpoint\": \"$C2F_STUDY_DIR/E3-seed$seed/step-$until.pt\"}" > "$spec"
        printf '\n=== %s: validation seed %s at %s ===\n' "$(date -u +%H:%M:%S)" "$seed" "$until"
        "$C2F_PYTHON" cluster/final_rehearsal.py --project "$C2F_PROJECT" --data "$C2F_DATA" \
            --runs "$spec" --output "$output" --device cuda
    done
done

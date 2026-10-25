#!/bin/bash -l
# Train E2/E3 with training seeds 0-2 on the frozen dataset-0.3 pilot corpus, then score them
# with the three E0 flow seeds on validation from one shared plan (cluster/final_rehearsal.py).
# The configs are the 17 October pilot's effective E2/E3 configs with only the seed changed.
# Usage: sbatch cluster/diffusion_repeats.sbatch. Finished runs are skipped; the evaluation
# folder must not exist. Test splits are never opened.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=c2f-diffusion-repeats
#SBATCH --output=diffusion-repeats-%j.out
#SBATCH --error=diffusion-repeats-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PROJECT:?}" "${C2F_PYTHON:?}" "${C2F_DATA:?}" "${C2F_STUDY_DIR:?}" "${C2F_E0_DIR:?}" "${C2F_SOURCE_REVISION:?}"
cd -- "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$C2F_PYTHON" -c 'import os,pathlib,click2future as m; assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(os.environ["C2F_PROJECT"]).resolve()/"src")'
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$C2F_SOURCE_REVISION"
mkdir -p "$C2F_STUDY_DIR"
runs=()
for conditioning in numeric spatial; do
    method=E2
    [[ $conditioning == spatial ]] && method=E3
    for seed in 0 1 2; do
        run="$C2F_STUDY_DIR/${method}-seed$seed"
        if [[ -e "$run/summary-5000.json" ]]; then
            printf 'Skipping finished %s\n' "$run"
        else
            printf '\n=== %s: train %s seed %s ===\n' "$(date -u +%H:%M:%S)" "$method" "$seed"
            "$C2F_PYTHON" -m click2future.training.train_diffusion --dataset "$C2F_DATA" \
                --run-dir "$run" --config "configs/training/diffusion_repeats/${conditioning}_seed$seed.json" \
                --device cuda --require-slurm
        fi
        runs+=("{\"method\": \"$method\", \"run_id\": \"$method-seed$seed\", \"checkpoint\": \"$run/checkpoint.pt\"}")
    done
done
for seed in 2 1 0; do
    runs=("{\"method\": \"E0\", \"run_id\": \"E0-seed$seed\", \"checkpoint\": \"$C2F_E0_DIR/flow-cvae-seed$seed.pt\"}" "${runs[@]}")
done
spec="$C2F_STUDY_DIR/runs.json"
(IFS=,; printf '{"sampling_steps": 32, "runs": [%s]}\n' "${runs[*]}") > "$spec"
printf '\n=== %s: validation evaluation ===\n' "$(date -u +%H:%M:%S)"
exec "$C2F_PYTHON" cluster/final_rehearsal.py --project "$C2F_PROJECT" --data "$C2F_DATA" \
    --runs "$spec" --output "$C2F_STUDY_DIR/validation" --device cuda

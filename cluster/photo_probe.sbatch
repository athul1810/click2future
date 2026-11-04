#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=c2f-photo-probe
#SBATCH --output=photo-probe-%j.out
#SBATCH --error=photo-probe-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
: "${C2F_PHOTO_ROOT:?Set the isolated photo workspace}"
cd "$C2F_PHOTO_ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export UV_CACHE_DIR="$C2F_PHOTO_ROOT/uv-cache"
export HF_HOME="$C2F_PHOTO_ROOT/hf-cache"
export HF_HUB_DISABLE_XET=1
export UV_HTTP_TIMEOUT=180
export HF_HUB_DOWNLOAD_TIMEOUT=180
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
uv_path="$HOME/.conda/envs/c2f-bootstrap/bin/uv"
repo="$C2F_PHOTO_ROOT/streamforce"
mkdir -p records
trap 'code=$?; printf "{\"exit_code\":%s,\"job_id\":\"%s\"}\n" "$code" "$SLURM_JOB_ID" > "records/exit-$SLURM_JOB_ID.json"' EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "records/hardware-$SLURM_JOB_ID.csv"
"$uv_path" python install 3.10
if [[ ! -x "$C2F_PHOTO_ROOT/.venv/bin/python" ]]; then
    "$uv_path" venv --python 3.10 "$C2F_PHOTO_ROOT/.venv"
fi
python_path="$C2F_PHOTO_ROOT/.venv/bin/python"
"$uv_path" pip install --python "$python_path" -r "$repo/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
"$uv_path" pip install --python "$python_path" gdown==5.2.0 psutil==7.2.2 \
    ninja==1.11.1.4 packaging==25.0 setuptools wheel
"$python_path" "$C2F_PHOTO_ROOT/source/cluster/fetch_streamforce.py" --repo "$repo" \
    > records/download.out 2> records/download.err &
download_pid=$!
# FA2 supports the A40; FA3 is Hopper-specific. Build against the pinned torch.
module load cuda/12.8.0
export MAX_JOBS=2
export NVCC_THREADS=2
"$uv_path" pip install --python "$python_path" flash-attn==2.8.3 --no-build-isolation
module unload cuda/12.8.0
wait "$download_pid"
"$uv_path" pip check --python "$python_path"
"$uv_path" pip freeze --python "$python_path" > "records/environment-$SLURM_JOB_ID.txt"
"$python_path" "$C2F_PHOTO_ROOT/source/scripts/run_streamforce_probe.py" \
    --repo "$repo" --output "$C2F_PHOTO_ROOT/probe-$SLURM_JOB_ID" \
    --seeds 0 1 2 3 --repeat-first

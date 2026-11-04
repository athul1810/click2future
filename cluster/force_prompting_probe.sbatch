#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=c2f-photo-fallback
#SBATCH --output=force-photo-%j.out
#SBATCH --error=force-photo-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PHOTO_ROOT:?Set the isolated photo workspace}"
node_root="/tmp/$USER/c2f-force-photo-20261104"
mkdir -p "$node_root/repo" "$node_root/source" "$C2F_PHOTO_ROOT/records"
chmod 700 "$node_root"
cd "$C2F_PHOTO_ROOT"
trap 'code=$?; printf "{\"exit_code\":%s,\"job_id\":\"%s\"}\n" "$code" "$SLURM_JOB_ID" > "records/exit-$SLURM_JOB_ID.json"' EXIT
cp -a force-prompting/. "$node_root/repo/"
cp -a source/. "$node_root/source/"
printf '{"host":"%s","root":"%s","job_id":"%s"}\n' "$(hostname)" "$node_root" "$SLURM_JOB_ID" > "records/node-$SLURM_JOB_ID.json"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export UV_CACHE_DIR="$node_root/uv-cache"
export HF_HOME="$node_root/hf-cache"
export HF_HUB_DISABLE_XET=1
export UV_HTTP_TIMEOUT=180
export HF_HUB_DOWNLOAD_TIMEOUT=180
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$node_root/cache"
uv_path="$HOME/.conda/envs/c2f-bootstrap/bin/uv"
"$uv_path" venv --python 3.10 --allow-existing "$node_root/.venv"
python_path="$node_root/.venv/bin/python"
"$uv_path" pip install --python "$python_path" \
    torch==2.5.0+cu121 torchvision==0.20.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
"$uv_path" pip install --python "$python_path" \
    diffusers==0.33.1 transformers==4.46.3 accelerate==0.34.2 \
    huggingface-hub==0.36.0 safetensors==0.7.0 tokenizers==0.20.3 peft==0.13.2 \
    numpy==1.26.4 pillow==11.3.0 sentencepiece==0.2.1 einops==0.8.1 \
    pandas==2.2.3 decord==0.6.0 controlnet-aux==0.0.9 mediapipe==0.10.21 \
    opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84 \
    imageio==2.37.2 imageio-ffmpeg==0.6.0 psutil==7.2.2
"$python_path" "$node_root/source/scripts/run_force_prompting_probe.py" \
    --repo "$node_root/repo" --output "$node_root/probe-$SLURM_JOB_ID" --seeds 0 1 2 3
mkdir -p "$C2F_PHOTO_ROOT/force-probe-$SLURM_JOB_ID"
cp "$node_root/probe-$SLURM_JOB_ID/"*.json "$node_root/probe-$SLURM_JOB_ID/"*.mp4 \
    "$C2F_PHOTO_ROOT/force-probe-$SLURM_JOB_ID/"
"$uv_path" pip freeze --python "$python_path" > "records/force-environment-$SLURM_JOB_ID.txt"

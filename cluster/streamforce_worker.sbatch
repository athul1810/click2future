#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=c2f-streamforce
#SBATCH --output=streamforce-worker-%j.out
#SBATCH --error=streamforce-worker-%j.err
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit with sbatch on the node holding the verified artifacts}"
: "${C2F_PHOTO_PROJECT:?Set the immutable uploaded project}"
: "${C2F_STREAMFORCE_ROOT:?Set the isolated environment and StreamForce workspace}"
: "${C2F_STREAMFORCE_PROBE:?Set the completed native probe JSON}"
: "${C2F_PHOTO_SPOOL:?Use a separate StreamForce spool}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="$C2F_STREAMFORCE_ROOT/hf-cache" HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR="$C2F_STREAMFORCE_ROOT/torch-compile"
export TRITON_CACHE_DIR="$C2F_STREAMFORCE_ROOT/triton"
"$C2F_STREAMFORCE_ROOT/.venv/bin/python" "$C2F_PHOTO_PROJECT/cluster/streamforce_worker.py" \
    --repo "$C2F_STREAMFORCE_ROOT/streamforce" --spool "$C2F_PHOTO_SPOOL" \
    --probe "$C2F_STREAMFORCE_PROBE" --raw-root "$C2F_STREAMFORCE_ROOT/raw" \
    --idle-seconds 1800

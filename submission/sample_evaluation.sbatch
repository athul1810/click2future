#!/bin/bash
#SBATCH --job-name=c2f-evaluation
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=tc2-evaluation-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${C2F_EVAL_PLAN:?Set the frozen evaluation plan}"
: "${C2F_EVAL_RUNS:?Set the checkpoint run specification}"
: "${C2F_SAMPLE_DIR:?Set a new generated sample folder}"
: "${C2F_RESULT_DIR:?Set a new result folder}"
: "${C2F_SOURCE_REVISION:?Set the source commit or source archive hash}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec .venv/bin/python - <<'PY'
import json
import os
import time
from pathlib import Path

import torch

from click2future.evaluation.sample_suite import generate_samples
from click2future.evaluation.suite import run_suite
from click2future.training.train_cvae import require_tc2

require_tc2()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)
started = time.perf_counter()
samples = Path(os.environ["C2F_SAMPLE_DIR"])
results = Path(os.environ["C2F_RESULT_DIR"])
plan = Path(os.environ["C2F_EVAL_PLAN"])
if results.exists():
    raise SystemExit("Use a new evaluation result folder")
record = generate_samples(plan, os.environ["C2F_EVAL_RUNS"], samples, device="cuda")
report = run_suite(plan, samples / "predictions.json", results)
job = {
    "status": report["status"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "source_revision": os.environ["C2F_SOURCE_REVISION"],
    "gpu": torch.cuda.get_device_name(),
    "torch_version": str(torch.__version__),
    "seconds_including_sampling_and_metrics": time.perf_counter() - started,
    "evaluation_plan_sha256": record["evaluation_plan_sha256"],
    "runs": len(record["runs"]),
}
(results / "job.json").write_text(json.dumps(job, indent=2) + "\n")
print(json.dumps(job))
PY

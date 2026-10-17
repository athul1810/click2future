# Click2Future source, Group 7

This archive contains the simulator, data generator, diffusion and CVAE trainers,
seeded samplers, auxiliary losses, evaluation, configs and correctness checks.
It contains no datasets or model weights. The diffusion pilots failed video-quality
checks; CVAE GPU training, final experiments and the demo remain unfinished.
This is not the final submission.

Use Python 3.12. Create a fresh environment and install the project from this folder:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -e '.[cpu,dev]'
.venv/bin/python submission/smoke.py
.venv/bin/python -m pytest -q
```

Use the matching CPU PyTorch distribution for CPU checks. TC2 training uses the
cluster's supported GPU environment. Dependency indexes can be configured outside
this archive through pip's normal environment or site configuration. The constraints
pin unambiguous versions from the working dependency lock. This is not an offline
dependency bundle. Package installation may need network access or a package cache.

The smoke command generates its own temporary two-variant dataset, validates replay,
loads a batch, and checks that every imported project module comes from this folder.
No existing dataset, checkpoint, GPU, or pretrained metric weights are needed.
On Windows, use `.venv\Scripts\python` for the Python commands.

Generate a grouped dataset with reproducible seeds and scene-level splits:

```sh
python -m click2future.cli generate-dataset --output-dir data/smoke --group-count 4 --variants-per-group 2 --seed 0 --config configs/simulator/smoke.toml
python -m click2future.cli validate-sample --input-dir data/smoke --sample-id scene-00000000-variant-000 --replay
```

The generator prints its manifest location. Read the generated dataset manifest for
the exact sample IDs. Keep generated data and all model files outside the submission.

For the controlled dataset-0.2, generate and validate its six action families:

```sh
python -m click2future.cli generate-controlled-dataset --output-dir data/controlled --group-count 10 --seed 0 --config configs/simulator/dataset_0_2.toml
python -m click2future.cli validate-controlled-dataset --input-dir data/controlled --replay
```

Both model families accept this dataset. Diffusion needs
`configs/training/pilot_diffusion_0_2.json` for its 1.1 wall-force limit. The older
config keeps its 0.5 limit. Both versions divide force by 0.5 without clipping.
The controlled pilot selects 21 clips from one training group; set `limit_train`
to null in a separate recorded configuration for a main run.

A separate dataset-0.3 corpus supports shape, texture, force and uncertainty
tests with seven disjoint scene-group splits. Train on that corpus before using
its held-out suites for matched generalization claims:

```sh
python -m click2future.cli generate-generalization-dataset --output-dir data/generalization --seed 200000 --groups-per-split 1
python -m click2future.cli validate-generalization-dataset --input-dir data/generalization --replay
```

This smoke generates 166 native clips. Use the Python generator's `group_counts`
argument to set a recorded larger split allocation. Both trainers support it;
the diffusion diagnostic is `configs/training/pilot_diffusion_0_3.json`. The
`force_limit` is 1.5 for the held-out forces, while only ID samples enter training.
Evaluate with explicit `--split id_test`, `shape_ood`, `texture_ood`, `force_ood`
or `physics_test`. All datasets and weights remain outside the submitted ZIP.

Evaluation modules provide `--help` for their input contracts:

```sh
python -m click2future.evaluation.video --help
python -m click2future.evaluation.images --help
python -m click2future.evaluation.futures --help
python -m click2future.evaluation.aggregate --help
python -m click2future.training.check_auxiliary --help
```

Image metrics need the `image-metrics` extra. LPIPS also needs pretrained AlexNet
weights in the user's Torch cache. Weights are excluded from this archive. The smoke
check and the core simulator/trajectory tests do not require them.

Diffusion training and
sampling modules provide their own input contracts:

```sh
python -m click2future.training.train_diffusion --help
python -m click2future.training.sample_diffusion --help
```

For a CPU smoke run, generate a dataset with the simulator pilot config. Copy
`configs/training/pilot_diffusion.json`, set precision to `float32`, steps to 4
and base_channels to 8, and save it as `cpu-smoke.json`. This preserves the
32-frame data contract while reducing the correctness-check cost:

```sh
python -m click2future.cli generate-dataset --output-dir data/pilot --group-count 8 --variants-per-group 2 --seed 7 --config configs/simulator/pilot.toml
python -m click2future.training.train_diffusion --dataset data/pilot --run-dir runs/cpu-smoke --config cpu-smoke.json --device cpu
```

The training checkpoint is local and excluded from the submitted archive. For
sampling, provide a condition NPZ with uint8 `first_frame[H,W,3]`, binary
`selected_mask[H,W]` and physical float32 `force_xy[2]`. The sampler writes uint8
`videos[K,T,H,W,3]`, preserves frame 0 and records hashes, seeds and timing.
These commands establish execution, not good generated motion. Train and evaluate
the actual methods before preparing the final submission.

For actual diffusion training on TC2, install the compatible CUDA environment,
then submit from this source folder:

```sh
export C2F_DATA_DIR=data/main
export C2F_RUN_DIR=runs/E2-seed0
export C2F_TRAIN_CONFIG=configs/training/pilot_diffusion.json
export C2F_SOURCE_REVISION=YOUR_SOURCE_ARCHIVE_SHA256
sbatch submission/train_diffusion.sbatch
```

The pilot config deliberately selects four clips. Copy it, set `limit_train` to
null and record validation-selected settings before a main run. Set `conditioning`
to `numeric` for E2 and `spatial` for E3, keeping the other choices matched.
Use independent runs for each training seed. Resume with the same source/config
and `--resume runs/E2-seed0/checkpoint.pt`. The job requires one visible TC2 GPU.

Evaluate complete pre-generated sample sets with the batch evaluator:

```sh
python -m click2future.evaluation.suite prepare --dataset data/main --split test --output outputs/test-plan.json
python -m click2future.evaluation.suite run --plan outputs/test-plan.json --predictions outputs/predictions.json --output-dir outputs/test-results
```

To generate those samples, edit the local checkpoint paths in
`configs/evaluation/runs.example.json`, then run:

```sh
python -m click2future.evaluation.sample_suite --plan outputs/test-plan.json --runs configs/evaluation/runs.example.json --output-dir outputs/model-samples --device cpu
python -m click2future.evaluation.suite run --plan outputs/test-plan.json --predictions outputs/model-samples/predictions.json --output-dir outputs/test-results
```

The runner derives method settings and training seeds from checkpoints, checks their
dataset/split hashes, and uses the plan's four seeds for every case. It records full
configs, training steps, timing and output hashes. Changed checkpoints or videos
reject evaluation. Untrained checkpoints require an explicit correctness-fixture
flag and remain labelled in summary JSON; they are not trained-model evidence.

For scheduled CUDA sampling and evaluation on TC2, set `C2F_EVAL_PLAN`,
`C2F_EVAL_RUNS`, `C2F_SAMPLE_DIR`, `C2F_RESULT_DIR` and `C2F_SOURCE_REVISION`, then
submit `sbatch submission/sample_evaluation.sbatch` from the source folder. Use new
sample/result folders. It requests one GPU, four CPUs, 24G system memory and six
hours. Sampling and metrics do not train models. E4 exports RGB through this runner;
auxiliary-head scoring stays separate.

The prediction JSON declares `source`, `sampling_seeds` and a `runs` list. Each
model run needs `method`, `run_id`, `training_seed`, a local `checkpoint` path and
a `videos` mapping from every plan `sample_id` to its generated NPZ path. Use
E0/E2/E3 and the plan's seeds. Paths are relative to the prediction JSON.
The evaluator writes per-case reports and summary JSON/CSV. It rejects changed
reference/config hashes and incomplete coverage. It never trains or selects a model.

For controlled data, add `--actions-config configs/evaluation/actions.yaml` to
plan preparation. The same evaluation then writes `actions.json` and `actions.csv`
for zero-force drift and paired reversed-force response. The fixed window is the
first 0.5 seconds. Every hidden-physics reference must be contact-free; exclusions
are recorded before model scoring. Missing RGB and stationary outputs keep their
counts. Bootstrap intervals resample scene groups, with no interval for one group.
Separate simulator-label and rendered-reference scores calibrate low-force cases
that remain below the motion threshold. These references never condition sampling.
Selected-object-only changes are not measured by these existing action families.
Dataset-0.3 force-OOD has no zero-force cases, so omit this flag for that split.


The CVAE baseline has a scheduled TC2 workflow. Generate a dataset with at least
ten scene groups, using `configs/simulator/pilot.toml` for 32-frame clips. With a
compatible CUDA environment on TC2, set the dataset and output folders, then run:

```sh
export C2F_DATA_DIR=data/cvae-smoke
export C2F_RUN_DIR=runs/cvae-overfit
sbatch submission/train_cvae.sbatch --overfit-samples 2 --max-steps 500 --checkpoint-every 50
```

Recorded one-clip attempts use `cvae_one_clip_1500.yaml` and
`cvae_one_clip_5000.yaml`. Both failed sampled-video quality despite low posterior
reconstruction error. To reproduce the longer diagnostic from source:

```sh
python -m click2future.cli generate-dataset --output-dir data/one-clip --group-count 10 --variants-per-group 2 --seed 226000 --config configs/simulator/one_clip_diagnostic.toml
export C2F_DATA_DIR=data/one-clip
export C2F_RUN_DIR=runs/cvae-one-clip-5000
sbatch submission/train_cvae.sbatch --config configs/training/cvae_one_clip_5000.yaml --overfit-samples 1
```

These are diagnostic settings, not accepted final baseline checkpoints. Strict
CUDA determinism remains enabled; pooled summaries use equivalent slice means.

Training requires one visible GPU in the MGPU-TC2 partition. The process limit is
5.5 hours within a six-hour job. It saves weights, optimizer/RNG state, and the shuffle
cursor. Resume with the same options and `--resume runs/cvae-overfit/latest.pt`.
`best.pt` uses fixed-seed validation reconstruction plus beta KL; prior-sample quality
must be evaluated separately. Full GPU training/resume remains unverified.

```sh
python -m click2future.training.sample_cvae --help
python -m click2future.training.check_components --output outputs/component-checks.json
```

The sampler writes four seeded uint8 videos with the original first frame and an
artifact manifest. The component check runs CPU gradients and untrained samples,
without optimizer updates. All final trained results still need completion before
preparing the final code ZIP.

`MANIFEST.json` lists each included file and its SHA-256. The code ZIP is one of the
three required submission files, alongside `group_07_report.pdf` and
`group_07_contribution.pdf`.

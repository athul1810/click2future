# Click2Future source, Group 7

This archive contains the simulator, data generator, diffusion and CVAE trainers,
seeded samplers, auxiliary losses, evaluation, configs and correctness checks.
It contains no datasets or model weights. The current pilots do not establish
held-out video quality. Training and checkpoint continuation are verified; final
experiments and the demo remain unfinished.
This is not the final submission.

The bounded multi-scene pilot has fixed configs in
`configs/training/multiscene_pilot.json`. It prepares dataset-0.3, computes contact
class weights from training labels only, and trains E0/E2/E3/E4 on TC2. Set
`C2F_PROJECT`, `C2F_PYTHON`, `C2F_PILOT_ROOT` and `C2F_SOURCE_REVISION`, then use
`sbatch submission/train_pilot.sbatch`. Choose a new pilot directory for each attempt.
For separate CPU data preparation, run
`python -m click2future.training.multiscene_pilot --root outputs/new-pilot --prepare-only`.
The same module's `--train-only` option uses that prepared corpus inside an allocation.

After all training finishes, set `C2F_PILOT_RESULTS` to a new directory and submit
`submission/evaluate_pilot.sbatch`. It evaluates validation clips, zero/reversed
forces, changed selection, reference coverage, E4 heads and CVAE latent diagnostics.
The initial GPU evaluation uses SSIM. Supplement its saved validation videos on CPU:

```sh
python submission/pilot_image_metrics.py --dataset outputs/new-pilot/data --evaluation outputs/pilot-evaluation --output outputs/pilot-images
```

This verifies saved reference/video hashes, excludes frame zero and includes a
static-copy control. It requires the image-metrics extra and cached LPIPS weights.
It does not establish physical accuracy. The 1.5 GPU-hour
cap includes evaluation. Keep checkpoints, generated data and result archives outside
the submitted source package.

The reserved 16/32/64-step validation study uses the same E3 checkpoint and
16 fixed cases. After pilot evaluation, set `C2F_STUDY_RESULTS` and submit
`submission/study_sampler.sbatch`. It reads the parent validation plan from
`C2F_PILOT_RESULTS`, retains excluded cases and uses the same four sampling seeds.
Its 0.5-hour limit is part of the combined pilot budget.

The optional flow-decoder CVAE uses `configs/training/cvae_flow_pilot.yaml` and a
separate training folder. It warps observed frame-0 pixels rather than predicting
new RGB. Its zero initialization copies a static image, so action response still
needs verification. Select the config with `submission/train_cvae.sbatch`, then
evaluate the trained checkpoint with `submission/flow_evaluation.py` inside an
allocation. This follow-up keeps the original RGB baseline results separate.

`cvae_flow_motion_pilot.yaml` is a further supervised follow-up. Future object
masks and centre displacements supervise the predicted field through a loss;
they never enter the prior or generator inputs. Keep it separate from the ordinary
CVAE comparison. Select this config with the same training command and use a new
run/evaluation directory.

`submission/check_e4_collision.sbatch` checks E4 on one fixed training collision
clip. Set the pilot environment above and `C2F_E4_CHECK` to a new directory. It
trains 5,000 updates, samples four futures and exports actual final denoiser heads
separately from RGB scores. Its training-only results do not establish generalization.

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

To isolate the clicked object, derive a separate held-out intervention plan:

```sh
python -m click2future.evaluation.targets --plan outputs/test-plan.json --output-dir outputs/target-controls
python -m click2future.evaluation.sample_suite --plan outputs/target-controls/plan.json --runs configs/evaluation/runs.example.json --output-dir outputs/target-samples --device cpu
python -m click2future.evaluation.suite run --plan outputs/target-controls/plan.json --predictions outputs/target-samples/predictions.json --output-dir outputs/target-results
```

Each scene's forward force is applied to two different objects with the same image,
duration, hidden physics and four sampling seeds. Derived references stay outside
the training corpus, whose original manifest/split hashes remain the compatibility
checks. The new plan freezes parent inputs and generated references. Keep its
`SPLIT-selected-object` results separate from ordinary ID/OOD tables.
The additional targets JSON/CSV report click following, per-object response
contrasts, nonselected drift, stationary pairs and missing tracks. All reference
physics must be contact-free in the fixed 0.5-second window. Below-threshold
reference responses stay in calibration and are never used to drop scenes.
An optional `--force-magnitude 0.4` uses the same shared magnitude in both
conditions while preserving the source direction. Choose it on validation, freeze
it before test evaluation, and keep the original-force calibration in the record.


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

The foreground and boundary diagnostic configs retain the same one-clip training
setup. The boundary CVAE uses weight 32 with a two-pixel margin around future
objects. It passes the declared training gate with 244/248 object observations and
complete selected-object tracks for all four seeds. Four nonselected observations
still fail, so no sample has every object tracked at every future frame. Its one
validation condition has no tracked future objects. This is a tiny-overfit check,
not successful generalization or the final trained baseline.

To reproduce that training check on TC2:

```sh
export C2F_DATA_DIR=data/one-clip
export C2F_RUN_DIR=runs/cvae-one-clip-boundary
sbatch submission/train_cvae.sbatch --config configs/training/cvae_one_clip_boundary.yaml --overfit-samples 1
```

`foreground_weight` defaults to 1 and `foreground_margin` to 0. Both are immutable
run settings. Future masks and their spatial margin supervise the loss only and
never enter inference. Diffusion applies the same optional weighting to its
epsilon/velocity objective. Keep the settings matched across compared methods.

Training requires one visible GPU in the MGPU-TC2 partition. The process limit is
5.5 hours within a six-hour job. It saves weights, optimizer/RNG state, and the shuffle
cursor. Resume with the same options and `--resume runs/cvae-overfit/latest.pt`.
`best.pt` uses fixed-seed validation reconstruction plus beta KL; prior-sample quality
must be evaluated separately. TC2 training and exact continuation have been checked;
these checks do not establish generated-video quality.

```sh
python -m click2future.training.sample_cvae --help
python -m click2future.training.check_components --output outputs/component-checks.json
```

The sampler writes four seeded uint8 videos with the original first frame and an
artifact manifest. The component check runs CPU gradients and untrained samples,
without optimizer updates. All final trained results still need completion before
preparing the final code ZIP.

## Local interactive demo

After obtaining a trained checkpoint and generating the matching development data:

```sh
python -m click2future.demo.server --dataset data/one-clip --checkpoint "e3=runs/e3/checkpoint.pt" --device cpu
```

Open the loopback address `127.0.0.1:8765` in a browser. `--device cuda` enables
local GPU inference. TC2 requires a scheduled compute allocation. The app uses
development/validation presets, never final-test presets; arbitrary image uploads
are deferred. Four paired seeds and the offline sampler produce the videos.
Simulator references and cached model replays have separate labels. Checkpoints
and generated replay files are excluded from this archive. The demo currently
establishes interface functionality, not held-out quality or a final model ranking.

`MANIFEST.json` lists each included file and its SHA-256. The code ZIP is one of the
three required submission files, alongside `group_07_report.pdf` and
`group_07_contribution.pdf`.

## supervised object-trajectory variant

e5 is a separate pose-supervised trajectory diffusion model with rigid decoding
of observed object pixels. its neural inputs remain the first image, selected
mask and physical force. it requires exactly two separately visible objects,
64px rgb and 32 frames. future positions and checked replay angles supervise
training only; no simulator generates its inference-time motion.

inside a tc2 slurm gpu allocation, using the matching generated frozen corpus:

```sh
python -m click2future.training.train_trajectory --dataset data/frozen --run-dir runs/trajectory --config configs/training/trajectory_pilot.json
python -m click2future.evaluation.trajectory_pilot --dataset data/frozen --checkpoint runs/trajectory/best.pt --output outputs/trajectory-validation --device cuda --steps 16
```

the pilot evaluator accepts validation only and requires the image-metrics extra.
model source hashes must match the checkpoint. this variant improves validation
action response but still fails strict completeness/contact quality; it does not
establish a sota result. weights, generated data and internal bundles are excluded.

`submission/train_trajectory.sbatch` and `submission/evaluate_trajectory.sbatch`
provide the tc2 job wrappers. set the required paths/revision listed at their top.
the training wrapper uses `C2F_DATA_DIR`/`C2F_RUN_DIR`; evaluation uses the same
data plus `C2F_TRAJECTORY_CHECKPOINT`/`C2F_TRAJECTORY_RESULTS`. use new output
directories and keep attempted runs. install dependencies before submitting.

## Contact and uncertainty audits

After the E5 validation run, replay its exact checkpoint and seeds inside TC2:

```sh
python -m click2future.evaluation.trajectory_audit --plan outputs/trajectory-validation/plan.json --predictions outputs/trajectory-validation/validation-samples/predictions.json --dataset data/frozen --checkpoint runs/trajectory/best.pt --output outputs/geometry-audit --device cuda
```

`submission/audit_trajectory.sbatch` declares a five-minute GPU allocation and
the required paths. Every saved RGB pixel must match before pose geometry is
scored. Oracle-motion renders calibrate the diagnostic only; they are never
model predictions. Raw overlap and approximate boundary counts are not accepted
physical-validity scores.

The CPU report audit retains all action families, null coverage and late drift:

```sh
python -m click2future.evaluation.uncertainty_strata --comparison outputs/trajectory-validation/validation-metrics/comparison.json --plan outputs/trajectory-validation/plan.json --predictions outputs/trajectory-validation/validation-samples/predictions.json --dataset data/frozen --checkpoint runs/trajectory/best.pt --output outputs/uncertainty-audit.json
```

The full-horizon stationary check supplements the unchanged half-second control.
Use fresh output paths. These commands measure failures rather than correct
sampled motion. Trained weights, data, raw pose arrays and measured working
artifacts remain outside this source ZIP.

## contact/stationary training candidates

e6 uses the same trajectory network and sampler with contact/stationary losses.
this is extra training supervision, not an inference correction. its checkpoints
use trajectory format 2. the original e5 checkpoint and format remain supported.

inside a tc2 allocation, use a fresh run directory:

```sh
python -m click2future.training.train_contact --dataset data/frozen --run-dir runs/contact --config configs/training/contact_token_pilot.json
python -m click2future.evaluation.trajectory_pilot --dataset data/frozen --checkpoint runs/contact/best.pt --output outputs/contact-validation --device cuda --steps 16 --method E6
```

an optional, separately hashed action supplement uses only existing training
worlds and physics. both selected objects receive the same eight directions and
two force magnitudes. original validation and final tests remain unchanged:

```sh
python -m click2future.data.action_supplement --parent data/frozen --output data/train-actions
python -m click2future.training.train_contact --dataset data/frozen --supplement data/train-actions --run-dir runs/contact-grid --config configs/training/contact_grid_pilot.json
```

these combined training changes are separate experiments, not isolated ablations
or accepted replacements. `submission/train_contact.sbatch` supplies the guarded
training wrapper. generate the supplement in a compute allocation too.

score full-horizon passive rest separately from the original early action checks:

```sh
python -m click2future.evaluation.passive_stationarity --plan outputs/contact-validation/plan.json --predictions outputs/contact-validation/validation-samples/predictions.json --dataset data/frozen --checkpoint runs/contact/best.pt --output outputs/contact-validation/passive-rest.json
```

all hidden-physics references must leave the passive object stationary and
contact-free. incomplete generated tracks remain missing rather than passing.

Passive-stationarity metric version 2 counts excursion at least one pixel as
drift, matching the shared motion protocol. Earlier strict-greater-than reports
remain historical results. For the 30-minute grid configuration, request a
40-minute allocation with `sbatch --time=00:40:00 submission/train_contact.sbatch`
and pass its grid config/supplement arguments. The default wrapper time is for
the shorter first recipe.


## Diagnose incomplete wall videos

After the pixel-exact pose audit, use its saved pose directory to measure canvas
clipping on CPU. For an E6 validation run, use matching E6 paths throughout:

```sh
python -m click2future.evaluation.wall_clipping --plan outputs/contact-validation/plan.json --predictions outputs/contact-validation/validation-samples/predictions.json --dataset data/frozen --checkpoint runs/contact/best.pt --poses-dir outputs/contact-geometry --output outputs/contact-wall-clipping.json
```

Create `outputs/contact-geometry` with the trajectory audit command above using
that same E6 plan, prediction manifest and checkpoint. The diagnostic verifies
every saved wall video against its poses, then compares native sprite area with
an unclipped padded view for each missing observation. It rejects truncated
measurement canvases. Original video metrics and missing observations remain
unchanged; the padded view is never used for model inference or replacement
scores. This is a clipping diagnosis, not a calibrated wall-contact accuracy.

## Diagnose collision errors and plan evaluation work

The validation collision audit retains every saved sample, matches all pose
renders to the original RGB, reports penetration severity/duration, and checks
the pixel collision estimator on every reference's own rendered video. It keeps
all hidden-physics reference comparisons without choosing a best reference.
Metric code and configuration must match the historical plan. It neither changes
official scores nor permits final-test tuning.

```sh
python -m click2future.evaluation.collision_audit --plan outputs/contact-validation/plan.json --predictions outputs/contact-validation/validation-samples/predictions.json --dataset data/frozen --checkpoint runs/contact/best.pt --poses-dir outputs/contact-geometry --config configs/evaluation/pixel.yaml --output outputs/collision-audit.json
python -m click2future.evaluation.inventory --dataset data/frozen --protocol configs/evaluation/comparison.yaml --output outputs/evaluation-inventory.json
```

The inventory reads only the dataset index, split manifest and comparison
protocol. It counts reference clips, conditions and proposed generated videos.
It does not open rollout arrays, certify reference eligibility, select a final
checkpoint or authorize test evaluation. Use fresh output paths for both tools.


## Matched wall-loss training comparison

The contact trainer supports an optional `boundary_weight` (default zero) and
`boundary_margin_pixels` (default one). It penalizes native-canvas excursion
during training only, using observed hulls and the existing low-noise gate. Old
contact checkpoints still load and resume when the added loss is disabled.

`contact_wall_control.json` and `contact_wall_pilot.json` differ only in that loss
weight. One allocated TC2 job regenerates a shared train-only action supplement,
trains both models from scratch, then evaluates every validation case and selected
object control with the unchanged 16-step sampler. Both recipes have 16,000
updates and a 20-minute process cap. The whole allocation is capped at 50 minutes.

```sh
sbatch submission/wall_comparison.sbatch
```

Set `C2F_PROJECT`, `C2F_PYTHON`, `C2F_DATA_DIR`, `C2F_WALL_RESULTS` and
`C2F_SOURCE_REVISION`. The output directory must be new. The driver retains
training records, validation reports, full-horizon zero/passive checks and saved
pose clipping diagnostics. It never evaluates final-test splits or changes
inference motion, tracking thresholds or the declared physics-quality gate.

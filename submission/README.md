# Click2Future source, Group 7

This archive contains the simulator, data generator, diffusion and CVAE trainers,
seeded samplers, auxiliary losses, evaluation, configs and correctness checks.
It contains no datasets or model weights. The final E0/E2/E3 runs and their
held-out evaluation (D014, plus the secondary D015 result) are complete, and no
method is a working physical model on held-out scenes. The `cluster/` scripts
described at the end reproduce those TC2 runs. The demo rehearsal remains open.
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

## CVAE reproduction, sweep and diagnostics

`submission/cvae_evaluation.py` applies the flow-follow-up validation protocol to
any trained CVAE checkpoint, RGB or flow; `flow_evaluation.py` stays unchanged for
its recorded results. Inside an allocation, pass the prepared pilot `--root`, a
trained `--checkpoint`, a new `--output` folder and `--device cuda`. Use the
β/latent configs in `configs/training/cvae_sweep/`, which change only `beta` and
`latent_dim` from `cvae_multiscene_pilot.yaml`.

`submission/cvae_force_probe.py` negates the force on nonzero-force validation
clips and decodes with fixed latents. It separates change through the condition
features from change through the prior mean, against prior seed spread and the
posterior-prior gap:

```sh
python submission/cvae_force_probe.py --dataset outputs/new-pilot/data --checkpoint runs/cvae/best.pt --output outputs/cvae-force-probe.json --device cuda
```

`submission/cvae_motion_timing.py` runs on CPU after an evaluation. It tracks every
frame of the saved samples of one suite and reports selected- and other-object
displacement at frames 4, 8, 16 and 31 against the simulator references:

```sh
python submission/cvae_motion_timing.py --plan outputs/cvae-eval/plan.json --data outputs/new-pilot/data --samples outputs/cvae-eval/validation-samples --output outputs/cvae-motion-timing.json
```

For the selected-object suite, pass `target-controls/plan.json`, its
`target-controls/references` folder and `selected-object-samples`. Future masks
choose scored regions and align identities only; they never enter the model. These
are validation diagnostics; they reinterpret the 0.5-second action controls but do
not replace them.
`submission/cvae_selection_drift.py` breaks the selected-object probe down per case
and seed: the other object's displacement, its motion along the force, onset frame and
minimum centre gap, plus the selected object per training action from clip metadata.

## Full-horizon action scores

`python -m click2future.evaluation.action_horizon --plan PLAN --predictions PREDICTIONS --output NEW.json`
reruns the unchanged action and selected-object scorers at the 0.5-second window
and at the final frame. A horizon is scored only where every hidden-physics
reference is contact-free. Plans without zero force, such as force-OOD, keep their
forward, reverse and reversal scores. The original window results stay unchanged.
It reads saved videos only; it never samples or trains.

Use `--relocate OLD=NEW` to read a saved TC2 plan's inputs from a local copy;
relocated files must keep their recorded hashes. Regenerating dataset-0.3 on a
different CPU architecture reproduced every checked rollout array, but one
geometry float in some metadata files differed in its last digit. Its manifest
hash then differs, and checkpoints trained on the original corpus refuse it.
Evaluate saved checkpoints against the corpus they were trained on.

## matched contact tail-risk refinement

`submission/contact_risk_comparison.sbatch` runs one bounded comparison. set its
required immutable source/interpreter/data/supplement/warm-start/output variables.
it reuses the retained training actions. the two recipes differ only in the
optional upper-tail penetration loss, with fresh optimizer and private rng from
the same ema checkpoint. this is explicit warm-start refinement, not resume or
an independent training repeat. data/label/geometry identities must match.

all three checkpoints, including the untouched anchor, are evaluated on validation
only. original metrics, passive rest, geometric replay, clipping and both action
horizons remain reported. the one-pixel margin and the public sampler are unchanged.
use fresh output directories; preserve failed phases. no final tests are opened.


## Experimental sampled-world relational refinement

E7 is a separate post-test exploratory comparison with training-only physical
labels and sampled prior worlds at inference. It does not replace the main
E0/E2/E3 results. It retains the existing force limit of 1.1.

Set `C2F_PROJECT` to committed source, `C2F_PYTHON` to the validated interpreter,
`C2F_DATA_DIR` to the frozen original corpus, `C2F_SUPPLEMENT` to the retained
training action supplement, `C2F_WARM_START` to the wall checkpoint,
`C2F_WORLD_RESULTS` to a fresh output and `C2F_SOURCE_REVISION` to the commit.
Submit `sbatch submission/world_comparison.sbatch` on TC2. The driver prepares
checked train/validation labels once, trains all three matched arms for 8,000
updates and evaluates the fixed final checkpoints plus the untouched anchor.
It preserves partial attempts. No test arrays are evaluated by this driver.

After an evaluation failure, the same driver accepts `--trained-run OLD_OUTPUT`
with a fresh `--output`. It verifies every completed training record, checkpoint,
recipe, teacher and cache identity before reusing the three fixed checkpoints.
It creates fresh evaluation plans and never trains again in this mode. Keep the
original comparison, failed logs and both source revisions; account for all jobs.

## Final TC2 runs (D013-D015)

These scripts produced the final runs and held-out results. Each needs
`C2F_PROJECT` (this source folder), `C2F_PYTHON` (the TC2 interpreter) and, where
listed, `C2F_DATA` (the dataset-0.3 folder) and `C2F_SOURCE_REVISION`. Use new
output folders. The training jobs skip finished stages, the evaluation jobs
refuse an existing output folder, and none overwrites results.

- `cluster/cvae_study.sbatch <config.yaml> ...` trains and validates flow CVAE
  configs into `C2F_STUDY_DIR`, using the prepared pilot in `C2F_PILOT_ROOT`.
  `configs/training/cvae_repeats/flow_seed1.yaml` and `flow_seed2.yaml` are the
  E0 seeds 1 and 2.
- `cluster/action_horizon.sbatch <run> ...` scores finished CVAE runs in
  `C2F_STUDY_DIR` at the 0.5-second window and the full clip.
- `cluster/diffusion_repeats.sbatch` trains E2/E3 seeds 0-2 from
  `configs/training/diffusion_repeats/` and scores them with E0 on validation.
  It also needs `C2F_STUDY_DIR` and `C2F_E0_DIR`.
- `cluster/final_rehearsal.sbatch` scores a run spec (`C2F_RUNS`) on validation
  from one shared plan into `C2F_OUT`. It never opens a test split.
- `cluster/final_evaluation.sbatch` freezes all ten test plans, then samples and
  scores every run in `C2F_RUNS` on the five reserved splits into `C2F_OUT`.
  `C2F_DECISION` labels the record (default D014).
- `cluster/diffusion_long.sbatch` trains E2/E3 for 40,000 updates with checkpoints
  at 10k, 20k and 40k (`C2F_STUDY_DIR`, seeds in `C2F_SEEDS`, default 0).

LPIPS needs the AlexNet weights in the Torch cache. Sampled predictions are bound
to plan hashes, and plans bind the evaluation source, so prepare, sample and score
from one account and one source revision.


## Temporal residual comparison

E8 freezes the conditioned E7 EMA and trains identical residual GRUs with state
reset at every frame or carried through time. Physical labels remain extra
training supervision. This is exploratory post-test work, not a main-method
replacement. Submit `submission/memory_comparison.sbatch` with immutable
`C2F_PROJECT`, validated `C2F_PYTHON`, verified `C2F_CACHE`, frozen `C2F_DATA_DIR`,
`C2F_WARM_START`, fresh `C2F_MEMORY_RESULTS`, and `C2F_SOURCE_REVISION`.
Each arm uses the fixed final update 4,000; partial runs and failed evaluations
are retained. `--trained-run OLD` uses completed checkpoints for fresh evaluation
without retraining. Do not overwrite original results or relax scoring margins.

## Paired visible-coverage comparison

The E7 architecture is unchanged. Two refinement arms append equally sized,
paired training data: old visible scenes with fresh physics, or independent
visible geometry/layouts with the same forces and physical draws. The common
contact sampling stratum uses only the union of paired training outcomes.
Original validation remains unchanged; this is an exploratory data intervention.

Submit `submission/coverage_comparison.sbatch` with `C2F_PROJECT`, `C2F_PYTHON`,
`C2F_DATA_DIR`, `C2F_CACHE`, `C2F_WARM_START`, fresh `C2F_COVERAGE_RESULTS` and
`C2F_SOURCE_REVISION`. Optional `C2F_PREPARED_COVERAGE` identifies a retained,
checked training-only generation; otherwise generation runs in the allocation.
The recipe is `configs/training/coverage_recipe.json`; both arms use
`configs/training/coverage_world.json` and fixed final update 8,000. The Python
driver supports `--trained-run OLD` with the same `--coverage` path for fresh
evaluation without retraining. Keep failed attempts and every declared seed.

## Matched surface messages

`submission/surface_comparison.sbatch` compares local vertex-to-segment messages
with pooled correspondence at the same parameter count. Both fully refine the
same conditioned E7 teacher on its original cache. Use `C2F_PROJECT`,
`C2F_PYTHON`, `C2F_DATA_DIR`, `C2F_CACHE`, `C2F_WARM_START`, a new
`C2F_SURFACE_RESULTS` and the immutable `C2F_SOURCE_REVISION`.
The two `configs/training/surface_*.json` recipes differ only in the local switch.
They use fixed final update 8,000, not validation-selected checkpoints.
The separately labelled E9 format-5 evaluation retains all four sample seeds,
geometry, contact, stationarity, action horizons, world interventions and
force/selection response. `--trained-run OLD` re-evaluates completed states in a
new output directory without retraining. This is exploratory reused validation,
not a fresh held-out improvement claim or a replacement for D014/D015.

## Prior-only decoder screen

`submission/prior_comparison.sbatch` compares a one-evaluation world-conditioned
pose decoder with the same E7 backbone retaining conditional diffusion. The
`configs/training/prior_*.json` recipes differ only in `world_prior_only`.
Use the same checked original world cache and teacher with `C2F_PROJECT`,
`C2F_PYTHON`, `C2F_DATA_DIR`, `C2F_CACHE`, `C2F_WARM_START`, a new
`C2F_PRIOR_RESULTS` and immutable `C2F_SOURCE_REVISION`. Final update 8,000 is
fixed. Direct inference uses one step; diffusion uses 16 Heun steps (31 calls).
Format 7 / E11 keeps this distinct from historical E7 and the team's E10.
This is a single-seed, reused-validation screen with extra physical supervision,
not a new held-out comparison. A deterministic fixed-world probe repeats the
same trajectory and labels those repetitions as dependent. Keep failed samples,
missing coverage, raw checkpoints and provenance; do not tune on reserved splits.

## Fixed-teacher repeat seeds

`submission/prior_repeats.sbatch` trains the two arms at refinement seeds 20/21
without opening any evaluation split. It checks completed seed-19 configuration,
training source and teacher identity. Provide `C2F_PROJECT`, `C2F_PYTHON`,
`C2F_CACHE`, `C2F_WARM_START`, `C2F_SOURCE_REVISION`, `C2F_SCREEN_SUMMARY` (the
completed direct screen summary) and a new `C2F_REPEAT_RESULTS`. Final update
8,000 is fixed. These are refinement repeats conditional on one E7 teacher,
not independent end-to-end training. Keep every state, summary and failed run.

The D018 inference driver is `submission/prior_holdout.py`. It requires all six
registered final-update checkpoint paths in `--runs`, the original corpus in
`--training-data`, and fresh `--dataset`/`--output` locations. Keep every attempt.
It scores only its registered ten-group ID corpus; it never opens D017.

From a Git checkout, commit the source before the run. From the source-only ZIP,
keep `MANIFEST.json` and all inventoried files unchanged. Run
`python -m click2future.submission verify-tree --root .` to verify the extraction.
The driver records the archive-manifest digest instead of inventing a Git commit,
and rechecks that identity at completion. Dependencies and generated outputs are
outside this source identity. Use the fixed training recipes to reproduce the
six checkpoints; the code ZIP contains no supplied weights or datasets.

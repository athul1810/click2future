# Click2Future source code, Group 7

This archive contains the simulator, dataset generator, model definitions,
training, sampling, evaluation, experiment configs and tests. The accompanying
report contains the measured results. Datasets, checkpoints, external pretrained
models, photographs and recordings are excluded. Reproducing trained results
requires generating data and training models; the CPU smoke check requires none
of those artifacts.

## Install and check

Use Python 3.12. On Linux or macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -e '.[cpu,dev]'
.venv/bin/python -m click2future.submission verify-tree --root .
.venv/bin/python submission/smoke.py
.venv/bin/python -m pytest -q
```

On Windows, create the environment with `py -3.12 -m venv .venv` and use
`.venv\Scripts\python.exe`. Use the environment's interpreter for later commands,
or activate it first. The smoke check generates two temporary rollouts, checks
replay, loads a batch and verifies that imports come from this extraction.
The two TC2-only tests require Linux. CUDA training requires an allocated GPU.

`constraints.txt` pins unambiguous dependency versions. GPU work needs the site's
compatible PyTorch build. Installation requires available packages or a cache.
SSIM/LPIPS evaluation needs the `image-metrics` extra; LPIPS also needs pretrained
AlexNet weights in the Torch cache. The smoke check does not need those weights.

`MANIFEST.json` lists every source file and its SHA-256. Verify the tree before
a registered comparison. Add generated data and outputs outside the inventoried
source directories; editing source changes its identity.

## Generate data

Generate a small grouped dataset and check deterministic replay:

```sh
python -m click2future.cli generate-dataset --output-dir data/smoke --group-count 4 --variants-per-group 2 --seed 0 --config configs/simulator/smoke.toml
python -m click2future.cli validate-sample --input-dir data/smoke --sample-id scene-00000000-variant-000 --replay
```

The manifest lists sample IDs. Variants of one visible scene stay in one split.
The controlled dataset adds zero, forward, reverse, two collision directions
and wall actions:

```sh
python -m click2future.cli generate-controlled-dataset --output-dir data/controlled --group-count 10 --seed 0 --config configs/simulator/dataset_0_2.toml
python -m click2future.cli validate-controlled-dataset --input-dir data/controlled --replay
```

The generalization dataset separates ID, shape, texture, force and repeated
physics. A small example creates 166 clips:

```sh
python -m click2future.cli generate-generalization-dataset --output-dir data/generalization --seed 200000 --groups-per-split 1
python -m click2future.cli validate-generalization-dataset --input-dir data/generalization --replay
```

Prepare the report's original seed-730000 corpus, with 16 training and four
validation scenes, using its fixed pilot recipe:

```sh
python -m click2future.training.multiscene_pilot --root outputs/pilot --prepare-only
```

This writes `outputs/pilot/data` and `pilot.json`. Keep that corpus for matched
runs. Regeneration on another platform can change metadata hashes even when
rollout arrays agree; saved checkpoints check their training-corpus identity.

## Reproduce the main comparison

E0 is the flow CVAE. E2/E3 are numeric/spatial pixel diffusion. The RGB CVAE and
E4 auxiliary heads are validation ablations. Inputs are frame-zero RGB, selected
mask and force; future labels enter training losses and evaluation only.

TC2 wrappers require `C2F_PROJECT` (this source directory), `C2F_PYTHON` (the
prepared GPU interpreter) and `C2F_SOURCE_REVISION` (the source/archive identity).
Set the extra paths named below and at the top of each script. Submit with
`sbatch`; wait for training to finish before dependent evaluation.

| Stage | Wrapper | Additional paths |
|---|---|---|
| RGB CVAE and E2/E3/E4 pilot | `submission/train_pilot.sbatch` | `C2F_PILOT_ROOT` |
| Pilot validation | `submission/evaluate_pilot.sbatch` | `C2F_PILOT_ROOT`, `C2F_PILOT_RESULTS` |
| Flow CVAE or KL/latent study | `cluster/cvae_study.sbatch CONFIG...` | `C2F_PILOT_ROOT`, `C2F_STUDY_DIR` |
| E2/E3 seeds 0-2 and validation | `cluster/diffusion_repeats.sbatch` | `C2F_DATA`, `C2F_STUDY_DIR`, `C2F_E0_DIR` |
| Primary five-split evaluation, D014 | `cluster/final_evaluation.sbatch` | `C2F_DATA`, `C2F_RUNS`, `C2F_OUT` |
| Longer diffusion training, D015 | `cluster/diffusion_long.sbatch` | `C2F_DATA`, `C2F_STUDY_DIR`, `C2F_SEEDS` |

For E0, pass these configs to the CVAE study wrapper:

```sh
sbatch cluster/cvae_study.sbatch configs/training/cvae_flow_pilot.yaml configs/training/cvae_repeats/flow_seed1.yaml configs/training/cvae_repeats/flow_seed2.yaml
```

Each writes `<C2F_STUDY_DIR>/<config-name>/train/best.pt`. Copy the three trained
checkpoints into `C2F_E0_DIR` as `flow-cvae-seed0.pt`, `flow-cvae-seed1.pt` and
`flow-cvae-seed2.pt`. The diffusion-repeat wrapper expects those names and writes
the combined `runs.json`. Pass that JSON as `C2F_RUNS` to the final evaluator.

Primary diffusion uses 5,000 batch-one updates; E0 uses 20,000 at batch two.
D015 extends diffusion to 40,000 updates, matching clips seen rather than GPU
time or optimizer updates. Set `C2F_SEEDS` to `0 1 2` for all three longer runs;
its default is seed 0. D015 was chosen after the primary test results.
`final_rehearsal.sbatch` is a validation evaluator; its filename does not imply
a required presentation.

Single-model interfaces provide training and sampling arguments:

```sh
python -m click2future.training.train_cvae --help
python -m click2future.training.sample_cvae --help
python -m click2future.training.train_diffusion --help
python -m click2future.training.sample_diffusion --help
```

Sampling conditions are NPZ files containing uint8 `first_frame[H,W,3]`, binary
`selected_mask[H,W]` and physical float32 `force_xy[2]`. Output is uint8
`videos[K,T,H,W,3]`, with the original first frame and a sampling record.

## Evaluate checkpoints

Edit local checkpoint paths in `configs/evaluation/runs.example.json`, then:

```sh
python -m click2future.evaluation.suite prepare --dataset outputs/pilot/data --split id_test --output outputs/id-plan.json
python -m click2future.evaluation.sample_suite --plan outputs/id-plan.json --runs configs/evaluation/runs.example.json --output-dir outputs/id-samples --device cpu
python -m click2future.evaluation.suite run --plan outputs/id-plan.json --predictions outputs/id-samples/predictions.json --output-dir outputs/id-results
```

Other split names are `shape_ood`, `texture_ood`, `force_ood` and `physics_test`.
Checkpoint paths are relative to the run JSON. Plans bind the dataset, settings
and scorer source; checkpoints and predictions must match them. Prepare, sample
and score with one source version, using fresh output paths.

For zero/reversed-force controls, add
`--actions-config configs/evaluation/actions.yaml` at plan preparation. Omit it
for force-OOD, which has no zero-force case. To vary only the selected object:

```sh
python -m click2future.evaluation.targets --plan outputs/id-plan.json --output-dir outputs/target-controls --force-magnitude 0.4
```

Sample and score the new `plan.json` with the same modules. Force 0.4 is the
validation-chosen selection probe. `click2future.evaluation.action_horizon`
rescores saved videos at 0.5 seconds and at the final frame. All references must
remain contact-free over the scored horizon. Missing tracks and excluded
conditions stay in the reports.

## Pose models and later experiments

E5 predicts rigid poses and renders observed object pixels. E6 adds contact,
stationarity and optional canvas losses. Both require two separately visible
objects, 64px RGB and 32-frame clips. Inside a GPU allocation:

```sh
python -m click2future.training.train_trajectory --dataset outputs/pilot/data --run-dir runs/trajectory --config configs/training/trajectory_pilot.json
python -m click2future.evaluation.trajectory_pilot --dataset outputs/pilot/data --checkpoint runs/trajectory/best.pt --output outputs/trajectory-validation --device cuda --steps 16
python -m click2future.data.action_supplement --parent outputs/pilot/data --output data/train-actions
python -m click2future.training.train_contact --dataset outputs/pilot/data --supplement data/train-actions --run-dir runs/contact-grid --config configs/training/contact_grid_pilot.json
```

Scheduled wrappers are `submission/train_trajectory.sbatch`, `train_contact.sbatch`
and `evaluate_trajectory.sbatch`. Evaluate a contact checkpoint with `--method E6`.
The paired-action supplement changes data and training budget, so E5/E6 is not
a loss-only ablation.

Later refinements use the E6 wall checkpoint, then an E7 teacher and checked
training cache. Wrapper headers give the required result variable and input paths:

| Experiment | Driver | Purpose |
|---|---|---|
| Matched canvas loss | `submission/wall_comparison.sbatch` | Train both E6 wall arms |
| E7 | `submission/world_comparison.sbatch` | Sampled physical properties and world-cache preparation |
| E8 | `submission/memory_comparison.sbatch` | Reset versus carried temporal state |
| Coverage | `submission/coverage_comparison.sbatch` | Matched narrow/broad data supplements |
| E9 | `submission/surface_comparison.sbatch` | Local versus pooled geometry messages |
| E10 | `submission/routing_comparison.sbatch` | Fresh numeric/spatial routing initializations |
| E11 | `submission/prior_comparison.sbatch` | One-pass versus diffusion decoding |
| D018 | `submission/prior_repeats.sbatch`, `prior_holdout.py` | Refinement seeds and new ten-scene ID evaluation |
| D019 | `submission/expected_prior_comparison.sbatch` | Analytic direct-loss expectation |

World refinements require `C2F_CACHE`, `C2F_WARM_START` and the original
`C2F_DATA_DIR`, besides the common source/interpreter variables. They share
the teacher's training history; their seeds are not independent end-to-end runs.

Later geometry and dynamics interfaces are:

```sh
python -m click2future.training.train_prior --help
python -m click2future.training.sample_trajectory --help
python scripts/compare_contact_projection.py --help
python scripts/compare_checkpoints.py --help
python scripts/run_physics_selection.py --help
```

The comparison driver covers energy scores, projection, loss scales and D038
components. D038 pairs `prior_rest_anchored_seed20.json` with
`prior_rest_anchored_dynamics_residual_seed20.json`; D039 pairs
`collision_impulse_screen_control.json` with `collision_impulse_screen_x1024.json`.
These configs are under `configs/training/`. The trajectory sampler's
`--momentum-block --contact-project` flags add explicit corrections. Omitting
both returns raw neural motion. Candidate selection runs only in its separate
experiment script, not in the ordinary four-seed sampler.

## Diagnostics

Use `python -m MODULE --help` or `python scripts/NAME.py --help` for these tools:

| Tool | Purpose |
|---|---|
| `click2future.evaluation.trajectory_audit` | Reproduce RGB from poses before scoring geometry |
| `click2future.evaluation.uncertainty_strata` | Coverage/spread by action, including missing ensembles |
| `click2future.evaluation.passive_stationarity` | Full-clip drift of reference-stationary objects |
| `click2future.evaluation.wall_clipping` | Native versus padded-render diagnosis of lost tracks |
| `click2future.evaluation.collision_audit` | Contact errors and reference-render calibration |
| `scripts/penetration_control.py` | Known-overlap control for hull depth |
| `scripts/pose_contact.py` | Pose contact timing with a training-chosen threshold |
| `scripts/metric_ceilings.py` | Simulator controls; the filename does not imply a mathematical bound |

Passive-stationarity version 2 counts exactly one pixel as drift, matching the
action protocol. The historical `submission/cvae_motion_timing.py` diagnostic
counts strictly above one pixel; it does not replace official action counts.
Exact simulator outlines in these audits are evaluation labels, not model inputs.

CVAE studies use `submission/cvae_evaluation.py`, `cvae_force_probe.py`,
`cvae_motion_timing.py`, `cvae_selection_drift.py` and the configs in
`configs/training/cvae_sweep/`. The 16/32/64-step study uses
`submission/study_sampler.sbatch`; E4 head inspection uses
`submission/check_e4_collision.sbatch`. These are validation or training
diagnostics, not additional final-test model selections.

## Optional interfaces

After generating development data and training a compatible checkpoint:

```sh
python -m click2future.demo.server --dataset outputs/pilot/data --checkpoint "e3=runs/E3-seed0/checkpoint.pt" --device cpu
```

Open `127.0.0.1:8765`. Synthetic uploads require a native 64x64 PNG with a flat
background, uniform border and two separated objects (at most 64 KiB).
Masks come from pixels; uploaded images
have no known simulator reference. Unsupported scenes are rejected. The
`--physics-baseline` mode uses an explicit solver with sampled properties,
labelled separately from neural predictions and ground truth.

The separate photograph interface starts with:

```sh
python -m click2future.demo.photo_server --spool outputs/photo --backend force-prompting --port 8770
```

Force Prompting/CogVideoX and StreamForce are externally pretrained backends.
This archive provides the interface, queue, SSH transport and worker adapters;
fresh generation also needs external model source, weights and a prepared GPU
environment. None of those artifacts is supplied. The `photo-remote` extra
supports SSH transport. Force Prompting returns 49 frames at 720x480 and 8 fps;
the native StreamForce path returns 81 frames at 832x480 and 16 fps with four
denoising steps.

Recorded replay needs a separately prepared spool of saved outputs. It is labelled
as replay and refuses uncached requests; fresh requests need an allocated worker.
Both interfaces are optional. Submit the report PDF, contribution PDF and this
source-code ZIP.

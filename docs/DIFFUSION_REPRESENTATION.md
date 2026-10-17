# RGB versus frame-0 residual diagnostic

The recorded single-clip velocity and boundary-weighted trials still fail usable
video quality. Test a different target before scaling the network or training set.

For normalized RGB future `x` and observed frame 0 `c`, the optional representation
is `r = (x - c) / 2`. Both RGB and residual targets are bounded by `[-1,1]`. Train
the same cosine DDPM objective on `r`, then sample with the same DDIM algorithm
and recover RGB as `clip(c + 2*r, -1, 1)`. Each DDIM step clips its x0 estimate in
RGB, so a residual stays within `[(-1-c)/2, (1-c)/2]` and both representations share
the same valid pixel range. Static pixels have target zero. This is
an invertible change of training coordinates before output clipping, not a motion
model, an external SOTA baseline or a claim of novelty.

The denoiser still receives only observed frame 0, selected-object mask and force.
Future RGB and future foreground masks remain training supervision. Neither is
available during inference. The future foreground weighting is unchanged from
the recorded boundary experiment; do not add a second loss change to this trial.

`TrainConfig.representation` defaults to `rgb`, including older checkpoints.
Saved residual checkpoints select their own decoding in the inference loader.
Exact resume rejects a representation change. Matched E2/E3/E4 comparisons reject
different representations; choose any new common setting on validation first.

## Fixed experiment

- One training clip from dataset-0.1, generated with seed 226000, 10 groups and two
  physics variants. This is the earlier diagnostic corpus, not the main benchmark.
- `configs/training/diffusion_one_clip_boundary.json`: 5,000 updates, seed 7,
  width 32, spatial conditioning, FP32 velocity, foreground weight 32, margin 2.
- Change only `representation`; repeat the RGB control from the same new source.
  Architecture, initialization, data order, timestep/noise draws, optimizer, LR
  horizon and stopping step are matched. No hyperparameter search.
- EMA samples, 32 DDIM steps and seeds 0-3. Report all attempts and hashes, training
  cost, future RGB MAE, tracking completeness and selected-object ADE/FDE.
- Include frame-0 copying as an explicit baseline. It can track static objects
  perfectly while failing motion; tracking alone cannot establish action response.
- Retain the existing training gate: at least 90% tracked future observations,
  complete selected-object trajectories and selected ADE at most 0.03125 on this
  64px corpus. A training pass still requires a separate held-out action pilot.

Submit `cluster/representation_probe.sbatch` with the same `C2F_PROJECT`,
`C2F_PYTHON`, `C2F_SOURCE_REVISION`, `C2F_DATASET` and `C2F_RUN_DIR` environment
variables as the earlier probe. Use a new run directory. If the dataset directory
is absent, the allocated job generates the fixed 20-clip diagnostic corpus.
Training and sampling execute inside Slurm; the login node only transfers files
and submits/checks jobs. The job reserves one GPU, four CPUs and 16 GB RAM for
45 minutes. The reservation is a cap, not measured runtime.

For a short plumbing check, the Python runner supports `--until` without changing
the configured LR horizon. Such a run cannot stand in for the 5,000-step trial.
It writes separate configs, training logs, checkpoints, RGB samples, previews and
`comparison.json`; a partial attempt instead retains `failure.json`.

Tests check residual round trips, zero residual for unchanged pixels, exact RGB
recovery with known denoising predictions, legacy loading, exact continuation for
both objectives/representations, observable-only sampling and matched-suite
rejection. Numerical and integration checks do not establish learned video quality.

## Measured result: 17 October

TC2 job 43235 completed both 5,000-update runs on an A40 with exit 0:0. The residual
hypothesis failed. Keep RGB as the default; neither run passes the existing gate.

| Representation | Tracked future observations | Complete all-object clips | Selected ADE | Future RGB MAE [0,1] |
|---|---:|---:|---:|---:|
| RGB | 124/248 | 0/4 | 0.00357, four complete selected trajectories | 0.02088 |
| Frame-0 residual | 70/248 | 0/4 | Undefined, selected trajectories incomplete | 0.03363 |
| Copy frame 0 | 248/248 | 4/4 | 0.04776, fails the motion threshold | 0.00275 |

These are four dependent samples on one training condition. They establish neither
generalization nor a method ranking. The copy baseline has lower image error while
failing selected motion. Training prediction errors are in different coordinates;
do not rank RGB and residual using their raw denoising losses.

The follow-up job 43238 checked EMA/raw weights and 32/128/1,000 DDIM steps with
seed 100. RGB stayed at 31/62 tracked observations in all six settings. Residual
ranged from 6/62 to 14/62. More sampling did not rescue this case. The two jobs
completed in 567 and 73 seconds, totalling 0.178 allocated GPU-hours. Checkpoints,
datasets and samples were backed up locally with verified SHA-256 hashes.

Both jobs ran from ff4f196, where the sampler clipped residual x0 to `[-1,1]` in
residual coordinates instead of to valid RGB. The sampler now clips in RGB. Training
and the RGB samples are unchanged by that fix, but the residual rows above used the
looser clip. Re-sample the saved residual checkpoint before treating the residual
result as final.

### What the missing counts mean

The nonselected object remains visible in the RGB samples, but every future
observation is rejected as fragmented/duplicated. All 3,448 candidate pixels
outside its largest component lie in the outer three-pixel border. Noisy border
colours resemble the dark object's frame-0 palette. Residual has the same problem
and additional area-change failures. The gate still fails; this diagnosis does
not turn the generated videos into successful outputs.

A separate control keeps every reference object pixel and trajectory unchanged,
then adds Gaussian noise only to future background pixels. With four noise seeds,
the unchanged tracker returns 248/248 observations at standard deviations 0, 2
and 4; 178/248 at 8; and 124/248 at 12 and 16, in uint8 intensity units. This
isolates a measurement limitation: background corruption can produce missing
object tracks even when the objects themselves are unchanged.

To reproduce that control, use the recorded training reference and seeds 0-3.
For each noise standard deviation, copy its RGB, draw
`np.random.default_rng(seed).normal(0, std, frames[1:].shape)`, round/clip to uint8,
and replace only pixels where
`~reference['instance_masks'][1:].any(axis=1)`. Assert all object pixels remain
exactly equal, then run the unchanged `TrackerConfig()` tracker on the corrupted
RGB and frame-0 masks. Future oracle masks construct this known control only;
they never enter tracking or model inference.

Next: improve appearance/border reconstruction and have the evaluation owners
calibrate extractor noise sensitivity on independent reference controls. Do not
relax thresholds against model test outputs. Any extractor change needs a new
version, retained negative controls and re-evaluation of all methods. Multi-clip
training and held-out action response remain open.

[Full scores, hashes and controls](results/diffusion-representation-checks.json).
The [sample panel](results/diffusion-representation.png) shows seed 0; all four
seeds are retained in the numerical record. Full source validation passed 406
tests and four subtests, plus lint, format and source-archive inspection.

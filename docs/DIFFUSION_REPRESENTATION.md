# RGB versus frame-0 residual diagnostic

The recorded single-clip velocity and boundary-weighted trials still fail usable
video quality. Test a different target before scaling the network or training set.

For normalized RGB future `x` and observed frame 0 `c`, the optional representation
is `r = (x - c) / 2`. Both RGB and residual targets are bounded by `[-1,1]`. Train
the same cosine DDPM objective on `r`, then sample with the same DDIM algorithm
and recover RGB as `clip(c + 2*r, -1, 1)`. Static pixels have target zero. This is
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

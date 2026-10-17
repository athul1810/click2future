# CVAE baseline and E4 integration

The CVAE model, training command, checkpoint handling, and prior sampler are implemented.
The shared batch adapter and joint diffusion/auxiliary objective are also implemented.
No generator has been trained by the CPU checks. TC2 overfitting, GPU resume, budgets,
and trained-sample comparisons remain required.

## CPU correctness

```sh
python -m click2future.training.check_components --output outputs/model-component-checks.json
```

This generates a two/three-object fixture, backpropagates through the real E4 noise
and auxiliary heads, exports heads in scene ID order, and checks CVAE gradients.
It also samples four untrained default CVAE videos at 64px with 32 total frames.
No optimizer updates occur. The default baseline has 574,659 parameters.
The record is in `results/model-component-checks.json`.

All 248 future object observations in the untrained CVAE example failed tracking.
The evaluator kept those failures instead of producing a good trajectory score.
This checks failure handling, not quality. The head export uses repeated teacher-forced
fixture outputs; it is not an E4 sampling result.

## Model and loss

The CVAE uses first-frame RGB, selected mask, and numeric force. A spatial encoder
feeds a learned diagonal Gaussian prior. A separate posterior sees future RGB during
training. A global clip latent and spatial condition feed the video decoder.
Sampling calls only the prior; the posterior and future encoder are never called.

Loss is future-frame mean RGB L1 plus beta times KL between posterior and prior.
Both terms average per clip; KL also averages over latent dimensions. Initial beta
is 0.001, with linear warmup over the first 20% of configured steps; log variances
are bounded to [-10,5]. Validation uses full beta throughout so the selection criterion
does not change during warmup. MSE is also logged and can be selected explicitly.
These settings match the baseline handoff and still need validation.
Pixel diffusion is adopted for the core comparison; optional autoencoder T08 is deferred.

`prepare_video_batch` converts loader RGB [0,1] to [-1,1], rearranges future frames
to [B,3,F,H,W], and divides force by 0.5 simulator units without clipping OOD forces.
It rejects invalid RGB/force and empty/nonbinary masks. Future masks, positions,
collisions, and hidden physics do not enter conditioning. Use the same frozen
normalization for comparable model runs.

The CVAE accepts dataset-0.1, controlled dataset-0.2 and dataset-0.3 manifests, with the
same split and file-hash checks. For matched comparisons, use the same dataset
revision for E0/E2/E3. Dataset-0.2 includes wall forces of magnitude 1.1, which
become 2.2 after normalization. A native wall condition passes the CVAE loader
and four-sample prior export in an untrained integration check. This does not
establish learned wall behavior. Test arrays remain outside the training index.
Dataset-0.3 is a separate matched generalization corpus; use its own training
and validation splits. See [its guide](GENERALIZATION_DATA.md).

## TC2 pilot

Generate a small native-format dataset. This is a smoke example, not the final dataset:

```sh
python -m click2future.cli generate-dataset --output-dir data/cvae-smoke --group-count 10 --variants-per-group 2 --seed 1000 --config configs/simulator/pilot.toml
```

Use the course TC2 environment with the project and a compatible CUDA backend installed.
The job requests one GPU, four CPUs, 24G system memory, and six hours in `MGPU-TC2`.
The trainer checkpoints at a 5.5-hour process limit and handles the early USR1 signal.
It refuses CPU training and missing/other partitions. Start with the overfit check:

```sh
export C2F_DATA_DIR=data/cvae-smoke
export C2F_RUN_DIR=runs/cvae-overfit
sbatch submission/train_cvae.sbatch --overfit-samples 2 --max-steps 500 --checkpoint-every 50
```

Check loss reduction, reconstructions, and four prior samples. Measure GPU memory,
step cost, and sampling latency before choosing the main budget. The config in
`configs/training/cvae.yaml` does not guarantee that the pilot fits TC2.

The recorded one-clip diagnostic now runs on the A40 after replacing adaptive
pooling with equivalent deterministic slice means. Five forward/gradient checks
cover divisible, nondivisible and overlapping bins. The 1500-step GPU run used
150,443,008 peak tensor bytes, and saved training states match across allocations.
The later 5000-step run still fails prior-video quality. These are small diagnostic
costs, not a full-corpus budget. See `results/cvae-training-checks.json`.

To reproduce the independent 5000-step diagnostic on TC2:

```sh
python -m click2future.cli generate-dataset --output-dir data/one-clip --group-count 10 --variants-per-group 2 --seed 226000 --config configs/simulator/one_clip_diagnostic.toml
export C2F_DATA_DIR=data/one-clip
export C2F_RUN_DIR=runs/cvae-one-clip-5000
sbatch submission/train_cvae.sbatch --config configs/training/cvae_one_clip_5000.yaml --overfit-samples 1
```

The 1500-step configuration is recorded separately. Preserve both failed quality
gates. All four prior samples of the single training clip are checked; examples
with missing tracks are retained. Use neither checkpoint as a final trained baseline.

`run.json` records source and dataset/split/content hashes, seed, config, parameter count, resume
history, elapsed time including IO/validation, and peak tensor allocation.
`metrics.jsonl` records per-step loss. Test arrays are never opened by the trainer.

Checkpoints keep weights, AdamW state, shuffle permutation/cursor, posterior/shuffle
RNG state, step, and best validation score. Resume requires identical code, config, data
bytes, and selected IDs. Keep the same output folder and options:

```sh
sbatch submission/train_cvae.sbatch --overfit-samples 2 --max-steps 500 --checkpoint-every 50 --resume runs/cvae-overfit/latest.pt
```

Main comparisons need independent initialization in a new folder. Resume continues
the same run and retains its inherited time. `best.pt` minimizes fixed-seed validation
posterior reconstruction plus beta KL over the configured validation subset; this
does not measure prior-sample quality. `latest.pt` supports continuation. Stopping
before the first validation may leave only `latest.pt`.

GPU optimizer continuation matched saved states exactly across TC2 allocations in
jobs 43165/43166. The resume operation's later metric export failed and was recovered
separately; see `results/cvae-training-checks.json`. Scheduler-signal interruption
still needs a live check. All model weights and generated data stay outside the code ZIP.

## Foreground-loss diagnostic

The seed-226000 training clip has only 2.36% object pixels. Low average image error
can therefore hide poor object reconstruction. Both trainers accept an optional
`foreground_weight`, defaulting to 1 so existing configurations keep their objective.
The new one-clip diagnostic configurations use 32, with the same 5000-step horizons,
model sizes, initial seeds and precision as the recorded unweighted attempts.

For each future pixel, the loss weight is 32 inside the union of valid object masks
and 1 elsewhere. Divide by the total weights, including RGB channels, within each
clip before averaging clips. CVAE applies this to L1 or MSE reconstruction; diffusion
applies it to epsilon/velocity prediction MSE. Uniform image errors remain in logs
separately. Reconstruction plus the declared KL still selects the CVAE validation
checkpoint; the diagnostic inspects the fixed-step latest checkpoint.

Future masks are training loss targets only. They never enter the model or sampler.
This is extra mask supervision shared by both model families, not an improvement
claim or the final selected training setting. Matched evaluation rejects checkpoints
with different foreground weights. Keep this setting identical across E0-E4 if it
is adopted, and retain the failed uniform-loss pilots. New loss weights require new
runs; exact continuation rejects an altered setting. Do not change RGB tracking or
trajectory thresholds to make the training gate pass.

The boundary diagnostic keeps weight 32 and adds `foreground_margin: 2`. It weights
the two-pixel neighbourhood as well as object interiors, so nearby background errors
also receive the increased penalty. Dilation is spatial only and clipped at image
edges. Margin 0 reproduces foreground-only weighting. Both the weight and margin
are immutable run settings and must match across compared methods. This follow-up
tests the enlarged-outline failure of foreground-only CVAE training, without changing
the model, seeds, step count, sampler or tracking thresholds.

TC2 job 43199 passes the predeclared CVAE one-clip gate: 244/248 tracked future
observations, selected-object ADE 0.00249 in normalized image coordinates, and
posterior MAE 0.00392. All four selected-object trajectories are complete. Four
nonselected observations fail, leaving zero fully complete all-object clips; those
failures remain reported. CPU prior inference on the complete one-condition
validation split still tracks 0/248 observations against each of its two hidden
references. Sampling is identical across those references, whose future labels
never enter inference. This verifies tiny overfit only. Full-data training and
held-out method comparisons remain open.

The matched E3 loss diagnostics still fail the same gate. EMA/32-step sampling
tracks 109/248 observations with uniform loss and 124/248 with either weighted
setting. The boundary setting has complete selected-object tracks and ADE 0.00348,
but the nonselected object's palette mask remains fragmented. Those missing tracks
remain failures. Both new TC2 jobs completed normally; together they used 0.3183
allocated GPU-hours, including sampling and reporting. Configs, seeds, raw counts
and sample strips are in [the diagnostic record](results/foreground-loss-checks.json).

## Sampling

Use an ID from the dataset manifest; the example ID does not imply a particular split:

```sh
python -m click2future.training.sample_cvae --checkpoint runs/cvae-overfit/best.pt --data-dir data/cvae-smoke --sample-ids scene-00001000-variant-000 --output-dir outputs/cvae-samples --run-id cvae-overfit --seeds 0 1 2 3
```

Sampling defaults to CPU; CUDA requires a TC2 allocation. Each NPZ contains uint8
`videos[K,T,H,W,3]` with frame 0 preserved exactly. `samples.json` keeps checkpoint,
video, and reference hashes plus seeds. Feed these videos to the existing evaluators.
Four different untrained outputs do not establish meaningful physical uncertainty.

## E4 handoff

`diffusion_objective` accepts the denoiser, prepared batch, noisy future, exact epsilon
target, timesteps, and schedule length. The trainer owns q(x_t|x_0), timestep sampling,
optimization, checkpoints, and reverse diffusion. The adapter requires epsilon prediction
and does not invent a schedule or sampler. See `training/check_components.py` for an
executable loader-to-backward example.

For E1-E3, omit auxiliary targets/config. For E4, pass `build_auxiliary_targets` and
`AuxiliaryLossConfig`. It combines noise MSE with the gated auxiliary term and records
component losses/counts. Labels reach only the loss, after the model call.
Training-only collision balancing remains in [AUXILIARY_LOSSES.md](AUXILIARY_LOSSES.md).

`evaluate_heads` accepts K batch-one outputs from actual sampler calls with their
timesteps. It restores scene IDs from frame-0 masks and exports `auxiliary_head`
reports. Features outside the declared supervision range are rejected.
`save_head_evaluation` writes the existing extracted-track NPZ and hashed JSON format.
Head and RGB scores must use the same checkpoint, conditions, seeds, and references
and remain separate. Actual E4 head-versus-video results still need trained samples.

`tests/test_diffusion_handoff.py` now runs against the integrated diffusion modules.
It checks identical native normalization and actual DDIM sampling with captured
timestep-zero heads for both epsilon and velocity targets. Capture preserves sampled
tensors and RNG state exactly and removes its hook after errors. These are untrained
inference checks without optimizer updates.
The initial integration snapshot passed 258 tests with no skips, recorded in
`results/diffusion-integration-checks.json`. The later batch-evaluation and issue-work
checks pass 322 tests; see `results/issue-work-checks.json`.

The standalone `diffusion_objective` helper remains epsilon-only. The integrated
diffusion trainer computes the selected epsilon/velocity target itself and applies
the same auxiliary loss. Do not pass a velocity target to the epsilon helper.
The standalone helper uses uniform pixel MSE; the optional foreground settings are
handled by the integrated trainer.

Use `with CaptureFinalHeads(model) as captured:` around each separate sampler call,
then pass `captured.output` to head evaluation. This uses actual generated-sample
features rather than a teacher-forced reference. Diffusion records an explicit
physical `force_limit` in its checkpoint: 0.5 by default and 1.1 in the dataset-0.2
pilot config. Its normalization remains force / 0.5. Keep the core model inputs
matched and predeclare the separate force-OOD range before comparisons.

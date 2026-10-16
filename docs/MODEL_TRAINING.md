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
Pixel diffusion is adopted for the core comparison; the optional autoencoder T08 remains open.

`prepare_video_batch` converts loader RGB [0,1] to [-1,1], rearranges future frames
to [B,3,F,H,W], and divides force by 0.5 simulator units without clipping OOD forces.
It rejects invalid RGB/force and empty/nonbinary masks. Future masks, positions,
collisions, and hidden physics do not enter conditioning. Use the same frozen
normalization for comparable model runs.

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

GPU optimizer resume and signal handling still need live TC2 checks. CPU tests cover
checkpoint serialization, synthetic optimizer moments, and RNG/cursor state without
generator updates. All model weights and generated data stay outside the code ZIP.

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
The combined suite passes 258 tests with no skips; the integration record is in
`results/diffusion-integration-checks.json`.

The standalone `diffusion_objective` helper remains epsilon-only. The integrated
diffusion trainer computes the selected epsilon/velocity target itself and applies
the same auxiliary loss. Do not pass a velocity target to the epsilon helper.

Use `with CaptureFinalHeads(model) as captured:` around each separate sampler call,
then pass `captured.output` to head evaluation. This uses actual generated-sample
features rather than a teacher-forced reference. B's current boundary rejects forces
above 0.5 simulator units; agree on the common force-OOD range before comparisons.

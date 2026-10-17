# CVAE handoff

Lead: Ananya, confirmed 17 October 2026. Athul and Ananya remain joint owners
of evaluation; agree their internal split before duplicating work.
Shrijeet owns the shared inference integration and demo. Finish by 14 November.

## Core implementation

Athul implemented the model, loss, TC2 trainer and prior sampler in PR #30.
Ananya takes the next CVAE training, prior-quality and validation work; this
ownership change does not reattribute the existing implementation.
See [the training guide](MODEL_TRAINING.md) for commands and correctness checks.
TC2 training, exact saved-state continuation and four-seed prior samples are now
measured in [the pilot record](results/cvae-training-checks.json). A CUDA pooling
backward crash is fixed without disabling determinism. The 1500-step and 5000-step
uniform-loss one-clip pilots failed sampled-video quality. The boundary-loss
follow-up in PR #37 passes the declared selected-object training gate: 244/248
tracked observations, selected ADE 0.00249 and posterior MAE 0.00392. All-object
completeness remains 0/4, and validation tracking remains 0/248. See
[the complete follow-up](results/foreground-loss-checks.json), including failures.
Athul's multi-scene run is now complete: 336 training clips, four validation
scenes and 24 conditions. The RGB prior tracks 398/5,952 observations with no
complete clips. Its validation posterior/prior MAE is 0.02091/0.02113 in [-1,1]
RGB units. Nonzero KL and RGB spread do not establish useful uncertainty.

A separate flow decoder tracks 5,654/5,952 observations and 80/96 complete clips,
but all eligible force and changed-selection pairs are stationary. It improves
appearance without passing the action checks. A further training-only displacement
loss uses future masks/positions as supervision, never as prior inputs. Keep that
supervised variant separate from the ordinary CVAE baseline. It tracks
5,765/5,952 observations, but passes only 1/16 force-reversal pairs and 0/16
selected-object pairs. Its validation-selected checkpoint is update 9,000 of
20,000 completed updates.
See [the measured pilot](MULTISCENE_PILOT.md) and
[the result record](results/multiscene-validation-checks.json) before another run.
Ananya retains CVAE ownership; these runs and code changes were carried out by Athul.

Use the same 64px/32-frame data, scene-group splits and observable condition
adapter as diffusion. Start with a compact conditional video VAE: encode
frame-0 RGB/mask/force into spatial features, infer one 64-dimensional clip
latent from the future RGB plus condition during training, and learn a
conditional diagonal-Gaussian prior from the condition alone. Decode the
latent with the condition into all 31 future frames, using temporal layers.
Append the supplied frame 0 exactly. Do not condition the prior on future
frames, future masks, hidden coefficients or a scene specification.

Start with mean RGB L1 reconstruction plus KL(q||p), both normalized explicitly
per clip. Use beta=1e-3 with linear warmup over the first 20% of training.
Compare beta in {1e-4, 1e-3, 1e-2} and latent size in {32,64} on validation
within a declared small budget, then freeze the choice. Check both motion and
posterior collapse; background reconstruction alone can reward static output.
Do not add oracle foreground conditioning to improve only this baseline.

## First handoff and acceptance

- Model/loss code, parameter count, tensor shapes and tests
  showing future inputs affect only the posterior, not prior sampling.
- A four-clip overfit attempt, four prior samples with fixed
  seeds, complete checkpoint/resume, and measured training/sampling cost.
- Held-out action controls, validation checkpoint selection,
  explicit posterior-versus-prior sample comparison and failure examples.
- Export the same uint8 `videos[K,32,64,64,3]` API as diffusion, preserving
  frame 0. Record config/data/checkpoint hashes, seeds, steps, precision and
  hardware. Demo and evaluator use prior samples only.
- Provide the baseline method text, cost table and actual contribution evidence.

If the first prior samples fail, keep that evidence and debug before running
the full suite. A posterior reconstruction is not a successful generative demo.

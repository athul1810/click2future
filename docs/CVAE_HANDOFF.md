# CVAE handoff

Proposed lead: Ananya, awaiting acceptance. Athul and Ananya remain joint owners
of evaluation; agree their internal split without leaving CVAE unstaffed.
Shrijeet owns the shared inference integration and demo. Finish by 14 November.

## Core implementation

The model, loss, TC2 trainer and prior sampler are implemented in PR #30.
See [the training guide](MODEL_TRAINING.md) for commands and correctness checks.
The TC2 overfit, GPU resume, trained samples and measured costs remain pending.

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

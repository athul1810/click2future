# prior-only decoder comparison

status: job 50455 completed and independently replayed. Neither arm is promoted. E11 / checkpoint format 7; format 6 belongs
to the team's separate E10 routing replication. This study does not open D017.

## question

E7 samples a physical world and also diffusion noise. A deterministic physical
system does not require the latter once its entire state is known. Our image
does not reveal that entire state: rasterization hides subpixel position and
shape, and the learned decoder may have model error. Removing diffusion noise
could therefore improve consistency or instead average distinct contact paths.
This is an empirical question, not a new general world-model capability.

Compare two refinements of the same conditioned E7 EMA. The control retains
conditional EDM. The direct arm gives the same backbone zero motion input and
noise level one and returns its clean-pose prediction in one evaluation. Its
only sampled variable is the existing independent physical prior. The same seed
keeps that draw fixed across actions. Actual input-scene physics is never read.

## fixed protocol

- original checked cache, 19,328 augmented training examples; unchanged validation;
- seed 19, batch 64, 8,000 updates per arm, final-update EMA only;
- identical transferred tensors, private RNG, sampling weights, AdamW schedule,
  physical-label supervision and contact/stationarity/canvas losses;
- identical sampled batches, noise levels and noise tensors; the direct arm
  ignores corruption inputs but retains their loss weights and activation rule;
- all parameters are in the optimizer, with identical nominal parameter count.
  Motion-input weights have zero direct-arm gradient. Effective capacity is
  constrained; this is not equal active capacity or equal gradient coverage;
- transferring E7 preserves the control function. It does not preserve the
  direct arm's sampled function. This is a warm-start decoder screen, not a
  demonstration that diffusion is unnecessary under arbitrary initialization;
- one direct evaluation versus 16 Heun steps / 31 evaluations for the control;
  do not call this matched inference compute or convert NFE into a speed claim;
- unchanged 24 validation conditions, all four seeds (96 videos), selected-object
  controls, geometric/pixel replay, native contact, horizon, uncertainty-family,
  mass-intervention and action-response diagnostics;
- no inference engine, projection, correction, reranking, seed omission or
  stationary-output override. Rigid decoding is the same observed-pixel decoder.

Fixed-world mass interventions repeat the same deterministic trajectory four
times in the direct arm. These are explicitly labelled dependent repetitions,
not four independent observations. Prior-predictive samples still use four
distinct sampled worlds. Training validation loss sees known world labels and
does not measure prior-predictive likelihood or calibration.

This is one exploratory training seed on four reused validation scenes, after
the main test evaluation. Neither an attractive clip nor a small aggregate gain
qualifies a checkpoint for promotion. Retain completeness, penetrations, worst
depth/duration, contact timing, passive drift, every action family, missing
coverage and all failed attempts. Any general quality claim needs independent
seeds and a separately registered untouched holdout. D014/D015 remain unchanged.

## execution

`submission/prior_comparison.sbatch` trains and evaluates both arms plus the E7
anchor. The driver rejects recipe differences beyond `world_prior_only`, checks
input hashes before and after, and rejects incomplete training. Both arms must
pass exact two-update CUDA continuation. Source, cache and teacher are frozen
before submission. A time-limited run is retained and cannot masquerade as an
8,000-update result.


## result, 3 november

Both arms finish 8,000 updates and complete all 96 primary clips. The direct
arm reduces penetrating clips from 11 to 1, penetrating frames from 105 to 4,
worst depth from 2.504 to 1.056 pixels and longest penetration from 19 to four
frames. Its 96-video sampling/export batch takes 2.31 seconds versus 31.06,
excluding checkpoint loading; this is not browser end-to-end latency.

Regressions remain: direct has two drifting zero-force clips out of 16, versus
none for diffusion, and timed-contact F1 falls from 0.3711 to 0.3107. Mean
selected-object ADE rises from 0.02466 to 0.02514. Seven trajectory cases fail,
versus ten for diffusion. Forward spread is 0.02498 versus 0.02346 and finite
reference spread 0.02580; that is descriptive, not probability calibration.
Both select the correct object in 64/64 eligible full-horizon pairs. Force
response completes 76/76 direct comparisons and 75/76 diffusion comparisons;
every complete comparison increases displacement. The missing comparison stays
missing. Eight selection and thirteen force conditions are excluded by the
same reference-contact eligibility rule.

All 926 collected file digests and 503 immutable source files verify. All 288
primary clips replay pixel-exactly in an independent Windows audit, with native
contact/event counts unchanged. The largest geometry-depth difference is below
2.79e-7 pixels. Both CUDA continuation fixtures are exact. The E7 anchor's 96
primary clips are identical to job 50260. Its native response counts keep their
platform-qualified boundary; they do not replace the earlier Windows counts.

This is a useful contact/cost tradeoff on four reused scenes, with one training
seed. It fails the rest and overall trajectory gates and is not an overall
quality win, independent replication, external comparison or promotion. The
complete record is `docs/results/prior-validation.json`; all attempts, model
states, samples and independent checks are retained in the private raw backup.

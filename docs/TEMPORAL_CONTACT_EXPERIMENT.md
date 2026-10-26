# temporal residual around frozen shared worlds

## question

E7's world-conditioned run restores forward spread and completes 96/96 clips,
but the worst penetration increases to 2.448 pixels and early reversal drops to
12/16. another attention block is not evidence of a contact model. this E8
experiment asks whether a learned residual benefits from carrying temporal state.

[Allen et al.](https://proceedings.mlr.press/v205/allen23a.html) demonstrate learned
rigid contact dynamics with general graph networks; explicit contact modules are
not a prerequisite for every task. [Act-FIGNet, v3](https://arxiv.org/abs/2509.12151v3)
uses action-conditioned mesh interactions and physical observations for contact
tasks. both differ from our single resting image. neither paper establishes that
a GRU on this small dataset will work. E8 is a bounded test of temporal residual
memory, not a reproduction of either method or an external SOTA comparison.

## implementation

the entire E7 conditioned EMA is frozen, including its physical-world encoder.
one GRU residual reads the predicted motion, first differences, other-object
motion, relative centres, bounding-box gaps, observable features, selected force,
sampled world and noise level. a zero-initialized output adds a learned EDM-scaled
residual. there is no solver, contact projection, clamping, output selection or
special zero-force override. only the new residual parameters receive gradients.
the frozen weights are hashed and checked at every saved state.

the two arms have identical tensors, active parameters, input features and losses:

| arm | GRU state |
|---|---|
| reset | zero state at every frame |
| carried | state passed through the 31 future frames |

both use the same GRU equations. the only recipe difference is `temporal_memory`.
each has 1,093,128 total parameters and 28,548 trainable residual parameters.
the residual is causal, but its frozen backbone attends to the entire noisy
future. this is not an end-to-end causal simulator or an explicit supervised
contact-state predictor. each denoising call starts with zero state; memory
cannot leak across calls, seeds or changed actions. worlds remain shared under
the existing per-seed prior contract.

## predeclared run

- teacher: conditioned E7 final update 8,000 from job 46889, checksum
  `93db6c714a0049b311117abe68faeef0d0e27080315bb8e67a2ecda59010c041`.
  it already inherited E6's wall EMA at update 15,500. this is refinement with
  additional physical supervision, not a scratch training comparison.
- reuse the verified original training/validation cache: 19,328 augmented
  training examples, 16 training geometries and four physical worlds per geometry.
  no new test data is opened. this remains post-D014 exploratory work.
- each arm: seed 19, batch 64, 4,000 updates, learning rate 1e-4, EMA 0.995,
  64-wide residual, same cosine schedule and existing losses. tail-risk weight
  stays zero. both inherit identical frozen teacher weights and initial residual
  tensors. the optimizer and private training RNG reset.
- evaluate fixed final update 4,000. known-world validation denoising loss is
  diagnostic only. incomplete training is retained and not treated as matched.
- evaluate the untouched E7 anchor and both arms with fresh validation plans,
  all 24 conditions, all four seeds and every hidden-physics reference. retain
  selection, both action horizons, full zero/passive rest, uncertainty strata,
  native exact RGB replay, clipping and overlap depth/duration. repeat the paired
  mass intervention on all eight eligible cases and four seeds.
- no metric or tolerance changes. judge depth/duration, contact timing, missing
  tracks, early action response and multireference coverage together. retain the
  original single-reference gate even where stochastic worlds make it difficult.
  a mean metric does not hide a failed case or a severe contact.
- one A40 allocation, four CPUs, 24 GB RAM, 45-minute cap. each training process
  has a ten-minute bound and preserves partial state. training runs only in Slurm.
  exact continuation is checked on the allocated device before either arm trains.
- preserve every attempt and both full states. code-only tests and frozen-weight
  checks are not learned-quality evidence. any useful candidate still needs
  independent training repeats, wider visible-world training and a newly frozen
  holdout before a confirmatory claim or default-demo promotion.

`submission/memory_comparison.sbatch` runs the comparison. required environment:
`C2F_PROJECT`, `C2F_PYTHON`, `C2F_CACHE`, `C2F_DATA_DIR`, `C2F_WARM_START`,
`C2F_MEMORY_RESULTS`, `C2F_SOURCE_REVISION`. all paths identify immutable source
and checked artifacts; the output must be new. `--trained-run OLD` on the Python
driver allows fresh evaluation after an evaluation failure without retraining.

E8 extends method/format guards and therefore needs fresh plans. historical E7
plans are replayed from `fb72774`; D014/D015 use the revisions recorded in
`FINAL_EVALUATION.md`. original scores and source archives stay unchanged.

## results

pending the declared TC2 allocation. no performance claim follows from this
implementation.

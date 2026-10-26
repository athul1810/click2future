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

job 47088 completed both 4,000-update arms and the full evaluation on an A40.
training and evaluation used immutable source `e19e7fa`. the later merge of
Athul's README encoding fix changes no model, training, sampling or evaluation
code. the fixed final checkpoints are retained; no intermediate checkpoint was
selected from these scores.

| measure | E7 anchor | reset | carried |
|---|---:|---:|---:|
| complete all-object clips /96 | 96 | 96 | 96 |
| penetrating clips /96 | 11 | 11 | 11 |
| penetrating future frames | 105 | 106 | 105 |
| maximum penetration, pixels | 2.44796 | 2.45162 | 2.45018 |
| longest consecutive penetration, frames | 19 | 20 | 20 |
| timed contact-event f1 | 0.33663 | 0.33333 | 0.33663 |
| mean selected-object ADE | 0.024742 | 0.024747 | 0.024738 |
| failed single-reference trajectory conditions /24 | 10 | 10 | 10 |
| forward selected-object spread | 0.022435 | 0.022446 | 0.022460 |
| forward reference-to-nearest-sample distance | 0.010764 | 0.010751 | 0.010764 |
| forward sample-to-nearest-reference distance | 0.009581 | 0.009576 | 0.009580 |
| correct early-window reversal pairs /16 | 12 | 12 | 12 |
| correct full-horizon reversal pairs /16 | 16 | 16 | 16 |

distances use canvas-width units unless marked pixels. the mean ADE is not the
per-condition gate; ten conditions still fail in every model. all three retain
0/16 full-horizon zero-force drift, 0/32 eligible passive drift, 0/16 incomplete
wall clips and 16/16 correct early selection targets and contrasts. forward
reference spread is 0.025803. paired mass interventions retain the expected
direction in 32/32 samples for every model; this behavior was already present in
the frozen teacher.

neither residual is promoted. carried memory does not improve contact counts or
timing and slightly worsens worst depth and duration. its paired centre-coordinate
change from E7 is only 0.00547 pixels RMS, with a maximum absolute coordinate
change of 0.02959 pixels. reset changes 0.01015 pixels RMS. both change RGB in
59/96 clips, but neither produces a useful visible improvement. these are effect
sizes over all saved predictions, not quality scores.

the frozen worst-contact condition
`g03-scene-00730016-collision_1_to_0-p00` remains the worst-depth condition in all
three models. every seed and all four reference profiles were inspected, as were
all seeds of the saved interactive push. the visual diagnostics agree with the
small numerical changes. no best seed is used to describe the candidate.

this bounds one small residual trained around a frozen teacher; it does not show
that temporal models cannot learn contact. teacher freezing, small residual
capacity, short adaptation and the narrow scene distribution remain possible
limits. the present evidence does not separate those causes. lower known-world
denoising loss would not establish improvement in generated contact dynamics.

## verification and retained evidence

- 622 tests and four subtests pass; lint and changed-source formatting pass.
  four inherited global-format failures remain. the corrected 285-file source
  ZIP passes inventory/link inspection, clean-extraction smoke and 16 extracted
  regression tests.
- all 773 collected file digests verify. all 458 committed training-source files
  verify before and after the allocation. both two-update CUDA continuation
  fixtures have equal losses and zero model/EMA differences.
- every frozen model and EMA backbone tensor, 82 per state, equals the original
  teacher EMA in both arms. only the residual was optimized.
- native and independent Windows replay are pixel-exact for 288/288 clips.
  contact counts and events match; the largest geometry-depth difference is
  2.79e-7 pixels. real-checkpoint demo adapters match offline pixels for every
  saved-push seed, and legacy E7 output is unchanged.
- Slurm records `COMPLETED`, exit `0:0`, 801 allocated GPU seconds and 2,242.30 MB
  maximum batch RSS. no jobs remain in the account after collection. the cuDNN
  warning about compacting copied EMA GRU weights is retained; the run completed
  within its declared cap. no performance optimization was inserted mid-run.

[results](results/temporal-memory-validation.json) retain every failed condition,
action stratum, intervention and checkpoint identity.
[checks](results/temporal-memory-checks.json) record source and asset digests.
the [private draft backup](https://github.com/athul1810/click2future/releases/tag/untagged-8edb44afd2ce55d79e71)
contains full states, the original teacher/cache, videos, poses, native and
Windows audits, analysis source, source archives and visual diagnostics. all five
release assets were downloaded again and their checksums verified.

## next experiment

expand independent training geometries and contact configurations before adding
another small adapter to the same teacher. keep the present cache as a control;
freeze new development geometry IDs and paired physical-world realizations
before training. separate increased data from architecture changes, and compare
the same backbone with matched update and sample budgets. include zero/passive
controls and collision timing as well as penetration in that comparison.

the existing evidence motivates this test but does not establish data coverage as
the sole bottleneck. any subsequent contact-state model needs its own matched
ablation. independent training repeats and a newly frozen holdout remain required
before a confirmatory gain or default-demo promotion. D014/D015 and these reused
validation scenes cannot serve that purpose.

# D018: prior-only replication, 3 november

Registered after the seed-19 screen and before generating or opening this
study's held-out corpus. The screen is exploratory; the new data must not be
used to change the architecture, loss, sampler or checkpoint rule.

## frozen training

Retain both seed-19 final-update models from job 50455, source 633669d. Train
both arms again with refinement RNG seeds 20 and 21. Every recipe differs from
the screen only in `seed`. Use the original 19,328-example world cache, identical
E7 teacher and training/model source. The driver rejects different source hashes
or teacher identity. All arms use final update 8,000, no validation selection.

These are three refinement seeds conditional on a single E7 teacher, not three
independently trained end-to-end systems. Direct input weights remain inactive;
equal nominal parameters do not establish equal active capacity. Both arms and
all attempted runs are retained. Training performs no held-out evaluation.

## separate held-out check

Generate a new corpus with generator seed 830000: ten ID-test scene groups and
one group in each other split required by the generator. Only the ID-test split
is opened for this study. Other splits remain unused. Compare all six fixed
checkpoints on every ID condition and all four existing sample seeds (0-3).
The generator, physics, scorer, tracking/contact thresholds, native resolution,
frame rate, samplers and force eligibility rules remain unchanged. Bind the old
training corpus and reject shared visible seeds or frame-zero pixels against
all its 668 clips. Freeze all plans and checkpoint hashes before sampling.

This corpus is separate from Athul's D017 seed 790000 and does not add E11 to
D017. It tests ID scene generalization, not shape/texture/force OOD. The original
D014/D015 and the four-scene E11 screen are retained as separate results.

## decision rule

Report every seed separately and scene-group aggregates, with all denominators
and missingness. A general advantage is not inferred from a pooled favourable
mean. Call an overall improvement only if each seed has:

- fewer penetrating clips and no worse worst depth/duration;
- no loss of complete clips, no additional zero-force or passive drift;
- no worse timed-contact F1, selected-object ADE and full-horizon action control;
- complete uncertainty-family accounting and no worse reference-to-sample
  coverage. Undefined coverage cannot pass the gate.

If any gate fails, report the contact/cost tradeoff and the failure. No inference
projection, rest override, reranking, seed removal or threshold change is allowed.
Sampling cost is measured on the same device and code path; NFE is not wall-clock
speed. Even a successful replication is a local result under extra physical
supervision, not an external state-of-the-art comparison or calibrated posterior.

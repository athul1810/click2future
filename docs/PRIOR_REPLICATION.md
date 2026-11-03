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
is model-scored for this study. Other required splits are generated and contract-validated,
but remain unscored. Compare all six fixed
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


Inference can run on the presentation laptop or in an allocated TC2 job; the
repeat training remains on TC2. Record the exact device, software and source.
The six final checkpoint hashes are resolved before any new-corpus generation.
The ID and selection plans are frozen before sampling. The inherited diagnostic
folder names include `validation-*`; their authoritative plan split is
`id_test`, with study ID D018 and explicit held-out scope in the result metadata.
No D017 data is generated, opened or scored by the D018 driver.

Gate details: selected ADE is the existing all-complete mean over conditions;
contact F1 is the existing micro score against the first declared reference.
Full-horizon zero-force/passive drift and action/force response retain the
existing one-pixel and reference-contact rules. Reference-to-sample distance is
checked separately for each action family's selected-object trajectories, with
all ten scene groups required. Undefined values fail the gate. No aggregate gain
can offset a regression in one of these predeclared checks.

## completed, 3 november

All six frozen models completed on ten new ID scenes using evaluation source
`16f2b3e85423d6790a74693cfbf13c8910fc2e06`. Training repeats used job 50537
(1,144 allocated A40 seconds, exit 0:0); inference used the RTX 3060 laptop.
The original preparation failure is retained: no plan/model score completed
before restoring 248 missing original-corpus NPZ files with native hashes.
The same new corpus, six checkpoints and gates were used on retry.

| seed | arm | complete | penetrating | max depth/duration | zero drift | F1 | ADE |
|---|---|---|---|---|---|---|---|
| 19 | One pass | 240/240 | 5 | 1.578/6 | 5/40 | 0.2745 | 0.03076 |
| 19 | Diffusion | 240/240 | 26 | 2.791/24 | 0/40 | 0.3053 | 0.02918 |
| 20 | One pass | 240/240 | 10 | 1.553/12 | 5/40 | 0.3015 | 0.03021 |
| 20 | Diffusion | 240/240 | 24 | 2.635/23 | 0/40 | 0.3191 | 0.02926 |
| 21 | One pass | 240/240 | 2 | 1.100/6 | 0/40 | 0.3005 | 0.02980 |
| 21 | Diffusion | 240/240 | 22 | 2.584/23 | 0/40 | 0.3118 | 0.02947 |

Penetration improves in all three refinement seeds, but every seed fails the
registered overall gate. ADE, contact F1, early force scaling and two coverage
families regress in all three seeds; rest also regresses in seeds 19/20. No
model is promoted. The frozen code also checks window response/reversal, as
committed before data generation; no threshold or gate changed after scoring.
All 1,440 primary pose replays, per-scene totals and 3,435 raw file digests
verify. D018 timing is observational with concurrent CPU checks; use the
separate quiet browser comparison for responsiveness.

See [full result](results/prior-replication.json). The later extracted-source
identity and reporting fixes do not change the actual evaluation revision.

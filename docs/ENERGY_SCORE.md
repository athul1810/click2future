# Multi-reference energy score (D031)

## Why

The strict trajectory gate compares each sample with the first reference only. D024 and
D030 show that the simulator itself, with true geometry and prior-sampled hidden
physics, fails that gate. A single reference cannot tell a good sampler of hidden
physics from a bad one.

Each controlled condition stores up to four references that differ only in hidden
physics. The energy score (Gneiting and Raftery, 2007) compares all samples with all
references of a condition:

    ES(X, y) = mean_i d(x_i, y) - 1/2 * mean_{i != j} d(x_i, x_j)

It is a proper scoring rule: in expectation it is lowest when the samples come from the
same distribution as the references. The pairwise term uses distinct pairs (the fair
form of Ferro, 2014) because each generator has only four samples. A condition's score
averages ES over its references; lower is better.

- Trajectory: displacement from frame zero for frames 1-31, in native pixels.
- Distance d: root mean square over frames of the Euclidean distance. The selected-object
  score uses the selected object; the all-object score sums both objects.
- Zero-force conditions have one reference.
- Code: `src/click2future/evaluation/energy_score.py`. Runner:
  `scripts/compare_checkpoints.py`.

The score is descriptive. It changes no gate, scorer, threshold or recorded result.

## Checks of the score itself

`metric_check` in `scripts/compare_checkpoints.py`:

- (a) An oracle that repeats the first reference four times has zero single-reference
  error and no spread. It must score worse than the prior-matched simulator.
- (b) No generator may be significantly better than the prior-matched simulator, using a
  paired scene-bootstrap 95% interval of the per-group difference.

A plain "the simulator has the lowest mean" rule was tried first, on original validation
with D020 only. The 0.15 px hull baseline scored 0.7066 against the simulator's 0.7222, a
difference inside noise for four scenes (interval -0.043 to 0.004). That rule was
replaced by (a) and (b) before any D032 or D033 run.

## First validation look (D020 only, original validation)

| Generator | Selected-object ES (px) | Single-reference selected ADE |
|---|---:|---:|
| Simulator, sampled hidden physics | 0.722 | 0.0233 |
| Physics baseline, 0.15 px bevel | 0.707 | 0.0239 |
| Physics baseline, 0.5 px bevel | 0.712 | 0.0239 |
| D020 | 0.806 | 0.0250 |
| D020 with projection | 0.802 | 0.0249 |
| Oracle, first reference repeated | 1.355 | 0.0042 |

The oracle is best by single-reference ADE and worst by energy score, as intended. D020
is worse than the simulator: the paired scene interval of the difference is 0.042 to
0.127 px, and most of the gap is in wall pushes (1.97 against 1.57 px). The new scorer
reproduces the committed D028 validation record exactly for D020 and its projection
(penetration, boundary, selected ADE, contact F1; same-world ADE to 1e-10).

## Registered runs

- D031: the opened seed-940000 D028 corpus, D020 with and without projection plus the
  four controls (`--mode d031`). It must reproduce the D028 numbers.
- D032 and D033 use the same score; see [GEOMETRY_LOSS_SCALE.md](GEOMETRY_LOSS_SCALE.md).

## Results (TC2 job 52537, source db539bc)

Record: `docs/results/energy-score-d031.json`. The first attempt, job 52502, stopped
before any score; see [GEOMETRY_LOSS_SCALE.md](GEOMETRY_LOSS_SCALE.md#runs).

**Reproduction.** The D028 corpus was regenerated in the job and its manifest equals the
D028 record (`4c65ceb1...`). The plan has the same 60 conditions, sampling seeds, input
file hashes and training-corpus binding; only absolute paths and the new
`energy_score.py` source hash differ. For D020 and its projection, every D028 scorer
field is equal: 19 and 0 penetrating clips, 25 and 0 boundary clips, selected ADE
0.02495 and 0.02476, contact F1 0.169 and 0.176, and the drift counts. Same-world ADE
agrees to 1.5e-10 (0.00706 and 0.00712).

**D031, opened D028 corpus** (ten ID scene groups, 60 conditions, 240 clips):

| Generator | Selected-object ES (px) | Minus simulator, 95% interval | Selected ADE |
|---|---:|---:|---:|
| Simulator, sampled hidden physics | 1.060 | | 0.0246 |
| Physics baseline, 0.15 px bevel | 1.062 | -0.009 to 0.016 | 0.0244 |
| Physics baseline, 0.5 px bevel | 1.071 | -0.012 to 0.039 | 0.0241 |
| D020 with projection | 1.103 | 0.020 to 0.068 | 0.0248 |
| D020 | 1.105 | 0.019 to 0.074 | 0.0249 |
| Oracle, first reference repeated | 1.304 | -0.056 to 0.519 | 0.0042 |

- `metric_check` passes. The oracle's mean is worse than the simulator's, although its
  interval includes zero on this corpus, and no generator is significantly better than
  the simulator.
- D020 is worse than the simulator, with the interval above zero. The projection removes
  every penetration and leaves the energy score almost unchanged (1.105 to 1.103).
- As on validation, wall pushes carry most of the gap (2.45 against 2.25 px). Collision
  conditions differ by 0.04 px and plain pushes by less than 0.01 px.

**Across all three corpora** (original validation, D028, D033):

- `metric_check` passes on each.
- The 0.15 px hull baseline is within noise of the simulator on each. Its intervals are
  -0.043 to 0.004, -0.009 to 0.016 and -0.004 to 0.026.
- Every learned checkpoint is significantly worse than the simulator on each corpus where
  it was scored, raw or projected. On D033 the D020 recipe's interval is 0.029 to 0.074,
  and the E10 routing models score 1.29-1.49 px against the simulator's 1.00.

The simulator draws from the reference distribution by construction, and the explicit
hull baseline cannot be told apart from it. The learned samplers are measurably worse,
most of all in wall pushes. The new D035 corpus shows the same pattern (see
[GEOMETRY_LOSS_SCALE.md](GEOMETRY_LOSS_SCALE.md)). The strict single-reference gate fails
the simulator itself;
the energy score places it with the two hull baselines at the top on every corpus, as a
proper score should.
It is still descriptive and changes no recorded result.

## Limits

- Four samples and at most four references per condition give a noisy per-case score;
  conclusions use scene-group intervals.
- All conditions use the plan's four sampling seeds, so samples share worlds across
  cases (the D025 limitation).
- The score uses pose trajectories, not pixels. It does not replace the RGB scorers.

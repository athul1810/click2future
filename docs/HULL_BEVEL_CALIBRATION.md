# D025-D027: contact geometry and calibration of the sampled futures

## D025: rerun of D021 with saved motions (TC2 job 52290)

The regenerated D021 corpus and the D020 final checkpoint match their recorded
hashes. Per-sample final travel was compared with the true simulator under the
same four prior draws. In free motion the hull baseline matches it exactly
(median ratio 1.000). In collisions the selected object travels about 7% too
short (median 0.93) and at walls about 1.4% too short. A rank test was also run,
but every case reuses the same four draws, so even the exact simulator is not
uniform there. That test is not a valid calibration check and is superseded by
D027. Record: `docs/results/calibration-d025.json`.

## D026: calibrated hull bevel (fit on train/validation, tested fresh)

`scripts/fit_hull_bevel.py` compares the hull baseline with the true simulator
under the same hidden physics on the original training and validation clips only.
Collision slop has no effect. The half-pixel bevel starts contacts early. The
lowest same-world selected ADE is at 0.15 px: 0.116 px versus 0.214 px at
0.5 px (`docs/results/hull-bevel-fit-d026.json`).

The protocol was committed at 4ed8636 before a new corpus (seed 930000) was
generated. TC2 job 52293 scored it with the unchanged D023 rules, 240 clips per
method, rejecting any frame-zero overlap with the training, D021 and D023 corpora.

| Fresh D026 ID | Hull 0.5 px | Hull 0.15 px |
|---|---:|---:|
| Complete clips | 240/240 | 240/240 |
| Penetrating / boundary-clipped clips | 0 / 0 | 0 / 0 |
| Rest / passive drift | 0/40, 0/80 | 0/40, 0/80 |
| Same-world displacement ADE | 0.00332 | 0.00177 |
| RGB contact F1 | 0.147 | 0.208 |
| First-reference selected ADE | 0.02206 | 0.02213 |

Same-world ADE fell in all ten scene groups; the paired bootstrap 95% interval is
[-0.00182, -0.00128]. The candidate fails one prespecified rule: first-reference
selected ADE is 0.00007 higher. It is rejected and 0.5 px stays the default. D024
suggests that prior sampling accounts for much of this single-reference measure, but the
rule was fixed in advance and is not changed after the result.
Record: `docs/results/hull-bevel-d026.json`.

## D027: calibration with independent draws (same job, fresh corpus)

For each of the 50 non-zero-force cases, 24 independent prior draws were
generated per method and the reference's final travel was ranked among them.

| Generator | Rank bins (5 each) | Uniform p | Reference outside all 24 |
|---|---|---:|---:|
| True simulator, sampled hidden physics | 14, 12, 9, 9, 6 | 0.43 | 0.04 |
| Hull 0.5 px | 10, 8, 8, 9, 15 | 0.49 | 0.20 |
| Hull 0.15 px | 15, 8, 6, 9, 12 | 0.29 | 0.20 |
| Neural D020 | 14, 9, 9, 12, 6 | 0.43 | 0.04 |

Expected outside fraction is 2/25 = 0.08. No generator rejects uniformity, but 50
correlated cases give little power. Every outside case for both hulls is a wall
push: the stopping point barely depends on hidden physics, so the 24 samples
cluster and sub-pixel shape error puts the reference outside. The neural model's
travel spread matches the exact simulator here, while its penetration failures
(D020/D021) remain a separate problem. Record: `docs/results/calibration-d027.json`.

These are bounded ID analyses with ten scene groups. No gate, scorer, default or
historical result was changed.

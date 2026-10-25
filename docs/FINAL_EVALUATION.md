# Final held-out evaluation

TC2 job 46716 evaluated the nine frozen runs from D014 on all five reserved
dataset-0.3 splits from one account. The runs are E0 flow CVAE, E2 numeric
diffusion and E3 spatial diffusion, each with training seeds 0-2. The setup:

- Source f003882 (#62), 32 DDIM steps and four sampling seeds per condition.
- All ten plans were prepared before any video was generated.
- Every suite completed, and nothing was tuned on these results.

The full record is [results/final-evaluation.json](results/final-evaluation.json).

To rescore, use source f003882 (D014) or fd43f11 (D015). Both have the same
`src/click2future/evaluation/` code. Each plan records the SHA-256 of every
file in that folder, so any later change there makes `read_plan` and
`action_horizon` refuse these plans by design.

## Tracking and image scores

Complete samples have every object tracked in every frame. SSIM/LPIPS are
single-reference scores pooled over the three training seeds.

| Split | E0 complete | E2 complete | E3 complete | E0 SSIM / LPIPS | E2 SSIM / LPIPS | E3 SSIM / LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| ID | 136/144 | 0/144 | 0/144 | 0.980 / 0.014 | 0.465 / 0.220 | 0.458 / 0.229 |
| Shape OOD | 90/144 | 0/144 | 0/144 | 0.966 / 0.029 | 0.475 / 0.248 | 0.466 / 0.244 |
| Texture OOD | 123/144 | 0/144 | 0/144 | 0.974 / 0.017 | 0.462 / 0.178 | 0.457 / 0.177 |
| Force OOD | 89/120 | 0/120 | 0/120 | 0.886 / 0.144 | 0.217 / 0.391 | 0.256 / 0.332 |
| Repeated physics | 132/144 | 0/144 | 0/144 | 0.973 / 0.012 | 0.457 / 0.180 | 0.451 / 0.183 |

E2 and E3 have no complete sample on any split, so their trajectory, collision
and action scores are undefined rather than zero. Selected-object ADE/FDE and
coverage are undefined for E0 too, because no scene group has all its cases
complete. E0's timed contact-event F1 is 0.09-0.22 across splits.

## E0 action controls

Correct reversal-sign rates are over complete pairs, with the exact-label and
rendered references in brackets. Zero-force drift counts complete samples.

| Split | 0.5 s window | Full clip | Zero-force drift, full clip |
|---|---:|---:|---:|
| ID | 0.08 (0.25 / 0.38) | 1.00 (1.00 / 1.00) | 0/24 |
| Shape OOD | 0.00 (0.25 / 0.25) | 1.00 (1.00 / 1.00) | 0/21 |
| Texture OOD | 0.04 (0.38 / 0.50) | 1.00 (1.00 / 1.00) | 0/24 |
| Force OOD | 0.00 (1.00 / 1.00) | not scored, contact first | no zero force |
| Repeated physics | 0.08 (0.31 / 0.31) | 1.00 (1.00 / 1.00) | 2/24 |

The validation pattern holds on held-out scenes. E0 moves the selected object in
the right direction over the full clip but too slowly for the 0.5-second window.

Force OOD shows the extrapolation limit. At 1.3/1.5 the references move clearly
inside the window. E0 moves all 24 reversed samples but none of the forward ones,
so it gets no reversal sign right.

Selected-object control fails on every split. No complete pair has the correct
target in the window. Over the full clip:

- ID has one contact-free group, and all eight of its complete pairs drift.
- Shape OOD has one contact-free group but no complete pair.
- The other splits reach contact first.

## What this supports

- The CVAE-versus-diffusion comparison favours E0 on every score defined for all
  three methods (tracking, SSIM and LPIPS), but
  E0 is not a working physical model. Its motion is slow, it does not scale with
  force and it does not respect the clicked object.
- The E2-versus-E3 comparison cannot be made on physical behaviour, because
  neither produces a trackable video. Their image scores differ by at most 0.04
  SSIM: E2 is slightly higher on four splits and E3 on force OOD. With two scene groups
  per split, the bootstrap intervals are not informative.
- These are measured failures under the declared protocol, with three training
  seeds per method. They are not tuned results.

## Follow-up after the test: longer E2/E3 training

This follow-up was run after the results above. It does not change them.

The final E2/E3 runs saw 5,000 clips in training (5,000 updates at batch 1),
against E0's 40,000 (20,000 updates at batch 2). TC2 jobs 46733 (seed 0) and
46788 (seeds 1-2) retrained E2/E3 for 40,000 updates with everything else
unchanged. Each seed was checked on validation at 10k, 20k and 40k. Seed 0:

| E2 / E3, seed 0 | Tracked of 5,952 | Complete of 96 | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| 5k (job 46704) | 176 / 5 | 0 / 0 | 0.456 / 0.451 | 0.167 / 0.167 |
| 10k | 2,353 / 1,895 | 0 / 0 | 0.819 / 0.830 | 0.072 / 0.073 |
| 20k | 3,389 / 2,930 | 4 / 0 | 0.908 / 0.919 | 0.050 / 0.049 |
| 40k | 3,434 / 2,869 | 7 / 5 | 0.931 / 0.933 | 0.045 / 0.047 |

The 5k SSIM/LPIPS values pool training seeds 0-2. Job 46804 then scored the
three E0 seeds and the six 40k runs together on validation, on A40 from one
plan. E0 reproduces job 46704 exactly.

| Validation, job 46804 | Tracked of 5,952, seeds 0/1/2 | Complete of 96 | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| E0 | 5,654 / 5,916 / 5,947 | 80 / 92 / 91 | 0.972 | 0.018 |
| E2, 40k | 3,434 / 3,479 / 2,726 | 7 / 3 / 0 | 0.929 | 0.049 |
| E3, 40k | 2,869 / 3,445 / 2,420 | 5 / 8 / 1 | 0.933 | 0.047 |

The full record is
[results/diffusion-long-validation.json](results/diffusion-long-validation.json).

- The noise in the 5k samples comes from the short training. At 40k the
  generated frames are clean, and SSIM/LPIPS close most of the gap to E0.
- Tracking still falls far short of E0. Most remaining failures are area
  changes and colour mismatches: objects change shape or shade during the clip,
  so the palette tracker loses them. In the inspected forward case the ball
  becomes squarer and darker by the last frames.
- E2 and E3 still cannot be separated. Their SSIM and LPIPS intervals overlap.
- The GPU model changes sampling. Seeds 1-2 were trained and first checked on
  an L40S node. Rescoring the same checkpoints on A40 moved tracked counts by
  49-208 and complete clips by up to 3. The table uses the A40 numbers.

## Secondary test result: E2/E3 at E0's clip count (D015)

D015 was declared after the D014 test results were seen and before this run.
These numbers are a secondary result. The D014 tables above stay the primary
comparison. D015 matches the number of training clips seen (40,000), not updates
or GPU time: E2/E3 take 40,000 updates at batch 1, E0 20,000 at batch 2, and
each 40k diffusion run took about 33 minutes on an A40.

TC2 job 46837 (source fd43f11, A40) ran E0 and the 40k E2/E3 runs, each with
training seeds 0-2, on all five reserved splits. It used the D014 setup:
32 DDIM steps, four sampling seeds, the same plans, scorer and force-OOD
handling. Every suite completed. E0 reproduces job 46716 exactly: all 522 E0
report files are byte-identical. The full record is
[results/final-evaluation-d015.json](results/final-evaluation-d015.json).

| Split | E0 complete | E2 complete | E3 complete | E0 SSIM / LPIPS | E2 SSIM / LPIPS | E3 SSIM / LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| ID | 136/144 | 0/144 | 10/144 | 0.980 / 0.014 | 0.933 / 0.057 | 0.933 / 0.059 |
| Shape OOD | 90/144 | 0/144 | 0/144 | 0.966 / 0.029 | 0.938 / 0.093 | 0.939 / 0.088 |
| Texture OOD | 123/144 | 11/144 | 15/144 | 0.974 / 0.017 | 0.937 / 0.049 | 0.937 / 0.048 |
| Force OOD | 89/120 | 0/120 | 0/120 | 0.886 / 0.144 | 0.744 / 0.188 | 0.788 / 0.155 |
| Repeated physics | 132/144 | 7/144 | 10/144 | 0.973 / 0.012 | 0.934 / 0.034 | 0.932 / 0.036 |

Per run, E2/E3 track 21-67% of object observations, against E0's 88-100%.
Selected-object ADE/FDE stay undefined for every method, and collision F1 is
undefined for E2/E3.

Action controls, as plain counts over the three training seeds (E0 / E2 / E3).
Reversal counts are correct pairs over complete pairs. Drift counts are
drifting samples over complete zero-force samples:

| Split | Reversal, 0.5 s window | Reversal, full clip | Zero-force drift, full clip |
|---|---:|---:|---:|
| ID | 2/24 / 0/16 / 0/16 | 24/24 / 1/2 / 0/0 | 0/24 / 0/0 / 0/8 |
| Shape OOD | 0/24 / 0/10 / 0/16 | 24/24 / 4/5 / 8/9 | 0/21 / 0/0 / 0/0 |
| Texture OOD | 1/24 / 0/24 / 0/21 | 24/24 / 2/3 / 2/2 | 0/24 / 2/6 / 0/11 |
| Force OOD | 0/24 / 0/0 / 0/0 | not scored, contact first | no zero force |
| Repeated physics | 2/24 / 0/9 / 0/12 | 24/24 / 0/0 / 0/0 | 2/24 / 6/6 / 7/8 |

Selected-object control fails as before. E2/E3 have up to 12 complete window
pairs per split. The only correct targets are E3's 2 of 11 on force OOD.
E2/E3 have no complete full-clip pair.

What this adds to the primary result:

- Trained on as many clips as E0, E2/E3 produce clean frames on every split. Their
  SSIM rises from 0.45-0.48 to 0.93-0.94, within 0.03-0.05 of E0. On force OOD
  it rises from 0.22-0.26 to 0.74-0.79, against E0's 0.89.
- Physical behaviour is still not shown. E2/E3 have 0-15 complete samples per
  split against E0's 89-136. In the 0.5-second window they get no reversal
  right (0 of 59 and 0 of 65 complete pairs). Over the full clip the few
  complete pairs mostly reverse correctly (7 of 10 and 10 of 11), but on
  repeated physics 13 of 14 complete zero-force samples drift, against 2 of 24
  for E0.
- Object identity is the failure. In an inspected ID forward case, both 40k
  models turn the small blue ball into a teal square by frame 16, so the
  palette tracker loses it. The 5k models leave heavy noise in the same case.
- E0 stays ahead on every score defined for all three methods (tracking, SSIM
  and LPIPS). E2 and E3 still cannot be separated. E3 has lower LPIPS on force
  OOD with intervals that just separate, but with two scene groups per split
  the intervals are not informative.
- So the D014 E2/E3 failure was largely an effect of too few training clips for image
  quality, but the main conclusion holds: no method is a working physical model
  on held-out scenes.

## Tracker failures by split

This table gives the share of future object observations the tracker loses,
pooled over training seeds. Selected-object missing rates follow the same pattern.

| Split | E0 | E2 5k | E3 5k | E2 40k | E3 40k |
|---|---:|---:|---:|---:|---:|
| ID | 2% | 86% | 87% | 47% | 48% |
| Shape OOD | 5% | 100% | 100% | 71% | 61% |
| Texture OOD | 3% | 92% | 93% | 48% | 48% |
| Force OOD | 6% | 97% | 99% | 73% | 69% |
| Repeated physics | 2% | 100% | 99% | 46% | 46% |

For E0 and the 40k models, new shapes and new forces raise tracker failures,
while new textures and repeated hidden physics stay close to ID. The 5k models
lose 86-100% everywhere.

## Cost

| | Parameters | Training | Training clips | Sampling per 32-frame video |
|---|---:|---:|---:|---:|
| E0 flow CVAE | 574,226 | 20,000 updates, 742-757 s | 40,000 | 0.012-0.015 s |
| E2/E3, 5k (D014) | 964,035 | 5,000 updates, 246-281 s | 5,000 | 0.54 s |
| E2/E3, 40k (D015) | 964,035 | 40,000 updates, 1,965-1,970 s | 40,000 | 0.58-0.59 s |

- Times are wall-clock on A40. E0 seed 0's GPU type is not recorded, and the
  40k seeds 1-2 trained on an L40S in 1,130-1,156 s.
- Sampling times come from the test jobs and include checkpoint loading and
  video export. E2/E3 run 32 denoising steps per video, so on average they
  sample 42 (5k) and 47 (40k) times slower than E0 in the same job.
- The final-phase TC2 jobs on Athul's account used 4.9 GPU-hours. TC2 states no
  total quota; jobs are capped at 6 hours and one GPU.

The record is [results/final-cost.json](results/final-cost.json).

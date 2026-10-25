# Final held-out evaluation

TC2 job 46716 evaluated the nine frozen runs from D014 on all five reserved
dataset-0.3 splits from one account. The runs are E0 flow CVAE, E2 numeric
diffusion and E3 spatial diffusion, each with training seeds 0-2. The setup:

- Source f003882 (#62), 32 DDIM steps and four sampling seeds per condition.
- All ten plans were prepared before any video was generated.
- Every suite completed, and nothing was tuned on these results.

The full record is [results/final-evaluation.json](results/final-evaluation.json).

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

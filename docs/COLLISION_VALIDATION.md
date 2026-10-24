# Collision validation and training handoff

The canvas-loss model still has 15/96 clips with hull penetration above the
existing one-pixel tolerance. All 15 are in the two contact families. Every
contact clip tracks completely, so visibility alone hides this failure.
The worst clip reaches 2.176 pixels and stays above tolerance for 22 consecutive
frames. The matched control has 15 such clips, maximum 2.216 pixels and a longest
run of 21 frames. Canvas loss improves wall visibility without fixing contact.

| Diagnostic | Matched control | Canvas loss |
|---|---:|---:|
| Penetrating clips /96 | 15 | 15 |
| Pair frames above one pixel /2,976 | 141 | 142 |
| Maximum hull penetration, pixels | 2.216 | 2.176 |
| Longest consecutive penetrating frames | 21 | 22 |
| Timed collision-event F1 | 0.3400 | 0.3232 |

The first worst-case example is
`g03-scene-00730016-collision_1_to_0-p00`, sample seed 2. For the canvas-loss
model, frames 10-31 remain above the tolerance. The result record retains every
penetrating clip, with its exact seed, frames and depth. Pixel-centre hulls are
an approximate geometry diagnostic, not continuous simulator penetration depth.
Oracle-motion controls remain at 0/84 clips above tolerance.

## What the timing score measures

The unchanged pixel contact estimator scores 0.4889 event F1 on all 84 validation
reference videos against their own exact labels: 22 matched, 24 extra and 22
missed events. All 84 timelines are complete. The model's 0.3232 is therefore not
a clean measurement of physical timing error alone. These are clean simulator
renders; the estimator's error on generated video may differ.

The 60 alternative hidden-physics reference videos score 0.2353 against the first
reference's labels, although they are valid simulator realizations. This also
shows why a single reference cannot specify the only correct event timeline.
Neither reference result is a performance bound or a replacement model score.
All per-reference comparisons are retained; no nearest reference is chosen to
improve the result. Existing published scores and thresholds stay unchanged.

The new audit verifies every saved pose against the original RGB, hashes all
inputs, and requires the historical collision configuration and metric source.
It records one-to-one event matches, unmatched events, missing intervals and
penetration duration for all 24 cases and all four seeds. Missing timelines stay
unscored and retain their missing counts.

## Next training check with Shrijeet

Use the retained contact failures to check the existing geometry objective and
train a candidate that removes sustained overlap. Preserve all 24 validation
conditions, four paired sample seeds, zero/passive rest, reversal and changed
selection checks. Report geometric depth/duration alongside the original timed
F1 and the reference-estimator calibration. A higher timing F1 alone is
insufficient. Keep exact labels and auxiliary-head scores separate from
video-derived contact estimates; future labels never enter inference.

Provide the checkpoint, config/source hashes, training seed, selected update and
full validation record. Use independent initializations for a loss comparison
and record warm starts separately. Reuse the existing data and exact paired
control when appropriate; do not start duplicate training for this audit.

Before the final force-OOD run, address the interface mismatch explicitly: the
current E6 config accepts force magnitude at most 1.1, while the existing recipe
uses 1.3 and 1.5. Changing the stored config after training or clamping test actions
would misrepresent that experiment. A reviewed OOD inference contract can preserve
the trained weights and normalization while declaring extrapolation; otherwise
report that suite as unsupported for this checkpoint.

The mismatch is specific to the 1.1 limit of E5/E6. The E2/E3/E4 configs declare
1.5, and the CVAE sampler divides force by 0.5 with no declared limit or clipping.
Every method still extrapolates in magnitude, because training uses one force per
action type:

| Action family | Training force | Force-OOD multiple at 1.3 / 1.5 |
|---|---:|---:|
| Forward and reverse push | 0.15 | 8.7 / 10.0 |
| Collision, both directions | 0.25 | 5.2 / 6.0 |
| Wall | 1.1 | 1.2 / 1.4 |

Report force-OOD results per action family rather than pooled.

## Reproduce the audit

```sh
python -m click2future.evaluation.collision_audit --plan outputs/validation/plan.json --predictions outputs/validation/validation-samples/predictions.json --dataset data/frozen --checkpoint runs/contact/best.pt --poses-dir outputs/validation/geometry-audit --config configs/evaluation/pixel.yaml --output outputs/collision-audit.json
```

Use the existing pose-audit directory for that exact checkpoint and ensemble.
This command reads validation only. It never trains, changes videos or relaxes
scoring. The compact [result record](results/collision-validation-audit.json)
contains the full failure list; private review artifacts retain detailed matches
and input hashes. Final tests and independent training repeats remain open.

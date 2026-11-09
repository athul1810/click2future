# Geometry-loss scale and the projection across checkpoints (D032, D033)

## Why

The penetration, contact-gap and canvas losses measure depth as a fraction of the image
width, where one pixel is 1/64, and then square it. In the D020 training log (job 50749,
last 20% of updates) the three weighted terms, after the expected objective's 0.84
activation factor, total about 0.004 of a 0.84 loss, about 0.5%. In D029 (job 52304) the
two added contact-dynamics terms were about 0.02% of the loss. D019,
D022 and D029 found no geometry gain. This study asks whether that is because the terms
were too weak. Gradients are clipped at norm 1.0, so larger weights change the update
direction, not its size.

## D032: scale sweep on original validation

Two families, each trained from the same E7 teacher and cache, seed 19, 8,000 updates:

| Family | Control | Candidates | Scaled weights |
|---|---|---|---|
| Rest-anchored (D020 recipe) | `rest_x1` = `prior_rest_anchored.json` | x16, x128, x1024 | penetration, contact gap, canvas |
| Direct with contact dynamics (D029 recipe) | `direct_x0` = `prior_direct.json` | `direct_dyn_x1` (D029), x128, x1024 | contact transition, normal velocity |

Configs: `configs/training/prior_rest_anchored_geometry_x*.json` and
`configs/training/prior_direct_contact_dynamics_x*.json`. Every other field equals the
base recipe.

Selection (`select_scale`, fixed before training). A candidate is eligible only if,
against its family control on original validation, raw outputs:

- it tracks the same number of complete clips;
- zero-force and passive drift do not increase, with no missing samples;
- selected ADE and same-world selected ADE rise by at most 0.0005;
- selected-object energy score rises by at most 0.032 px (0.0005 normalized);
- it has fewer penetrating clips.

Among eligible candidates, pick the fewest penetrating clips, then the fewest boundary
clips, then the smaller scale. If none is eligible, the family has no selection.

## D033: fresh confirmation

A new corpus, seed 950000, ten ID groups (`FRESH_COUNTS`), is generated after the rules are
committed. Frame-zero images are checked against the original, D021, D023, D026 and D028
corpora, and the plan is bound to the original training corpus.

Checkpoints: both family controls, any D032 selection, and the six E10 routing models
(numeric and spatial, seeds 19-21, `best.pt`). Every checkpoint is scored raw and with
the D028 projection (per-frame mode). Masses weight the projection for world-conditioned
models; E10 models do not see a world, so they use equal masses. Controls: simulator
with sampled hidden physics, first-reference oracle, and both hull baselines.

Rules (`scripts/compare_checkpoints.py`):

- `projection_holds`, per checkpoint: the projected run tracks the same complete clips, has
  zero penetrating and zero boundary-clipped clips, adds no drift, and stays within 0.0005
  selected ADE and 0.032 px energy score of the raw run.
- `learned_geometry_holds`, per D032 selection against its control, both raw: fewer
  penetrating clips, no more boundary clips, no new drift, selected ADE within 0.0005,
  paired scene-bootstrap upper bound of the same-world ADE change below 0.0005, and
  energy score within 0.032 px.
- `metric_check`, see [ENERGY_SCORE.md](ENERGY_SCORE.md).

Same-world ADE is only meaningful for models that receive the sampled world; for E10 it
is reported but not interpreted.

## Runs

The first attempt, TC2 job 52502 at 285d482, trained all eight arms and then stopped
before producing any score. The scorer asserted that zero force leaves every rendered
pixel unchanged, a check copied from the D028 script, where only the rest-anchored D020
model and the hull baseline were scored. Both are exact by construction: the rest-anchored
model subtracts its own zero-force response, and the baseline starts at rest. Direct and
E10 models have no such guarantee. The D029 direct models move on 2 of 16 zero-force
validation samples, which the zero-force drift count already records. Two fixes, made
before any score was seen:

- `zero_pixel_check` records how many zero-force samples change pixels for every
  pose-producing arm and requires zero only for rest-anchored checkpoints (raw and
  projected) and the hull baselines. It is reported, not raised, so a run is never lost.
- An undefined selected ADE (incomplete tracks) on either side now fails a rule instead
  of raising.

No rule, margin or arm changed. A local run on validation covering every model type
(D020, both D029 checkpoints, E10 numeric and spatial) reproduced the D028 values for
D020. The full job is repeated, keeping each checkpoint as soon as it trains. It also
compares the retrained `rest_x1`, `direct_x0` and `direct_dyn_x1` weights with D020 and
D029.

The second attempt, job 52537 at db539bc, completed every step on one A40 (TC2N01). All
eight arms trained for 8,000 updates. The retrained `rest_x1`, `direct_x0` and
`direct_dyn_x1` are bitwise equal to the D020 and D029 checkpoints in model and EMA
weights. `zero_pixel_check` passes in all three modes.

## Results

Records: `docs/results/geometry-loss-scale-d032.json` and
`docs/results/checkpoint-projection-d033.json`.

### D032: selection on original validation

Four scene groups, 24 conditions, 96 clips. "Share" is the scaled terms' share of the
training loss over the last 20% of updates. ES is the selected-object energy score in px.

| Arm | Share | Penetrating | Boundary | Selected ADE | Same-world ADE | ES | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| `rest_x1` | 0.5% | 4 | 6 | 0.02502 | 0.00734 | 0.806 | control |
| `rest_x16` | 2.1% | 2 | 2 | 0.02541 | 0.00781 | 0.837 | eligible, selected |
| `rest_x128` | 7.5% | 2 | 0 | 0.02527 | 0.00819 | 0.876 | same-world and ES margins fail |
| `rest_x1024` | 20.0% | 0 | 0 | 0.02691 | 0.01025 | 0.931 | ADE margins and ES fail; one passive drift |
| `direct_x0` | 0 | 1 | 6 | 0.02515 | 0.00763 | 0.832 | control; 2 zero-force drifts |
| `direct_dyn_x1` | 0.02% | 1 | 6 | 0.02516 | 0.00760 | 0.833 | penetration not lower |
| `direct_dyn_x128` | 0.9% | 1 | 6 | 0.02501 | 0.00728 | 0.823 | penetration not lower |
| `direct_dyn_x1024` | 3.4% | 0 | 6 | 0.02496 | 0.00668 | 0.819 | eligible, selected; no drift |

- Rest family: larger weights give fewer penetrating and boundary clips (4 and 6 at x1,
  0 and 0 at x1024), while same-world ADE (0.00734, 0.00781, 0.00819, 0.01025) and
  energy score (0.806, 0.837, 0.876, 0.931) rise at every step. The losses trade motion
  accuracy for separation. `rest_x16` passed with little room: 0.00003 on same-world ADE
  and 0.0004 px on energy score.
- Direct family: at x1 and x128 penetrating and boundary clips do not change. At x1024
  the single penetrating clip goes, and selected ADE, same-world ADE and energy score
  are all slightly lower. Its zero-force drift also falls from 2 samples to 0.

### D033: fresh confirmation

New seed-950000 corpus, ten ID scene groups, 60 conditions, 240 clips. Manifest
`32e9a852...`; no frame-zero image repeats one of the five earlier corpora.

| Checkpoint | Penetrating | Boundary | Zero-force drift | Selected ADE | Same-world ADE | ES |
|---|---:|---:|---:|---:|---:|---:|
| `rest_x1` (D020 recipe) | 9 | 22 | 0 | 0.02689 | 0.00709 | 1.055 |
| `rest_x16` | 5 | 8 | 0 | 0.02694 | 0.00734 | 1.071 |
| `direct_x0` | 3 | 15 | 5 | 0.02701 | 0.00734 | 1.089 |
| `direct_dyn_x1024` | 0 | 14 | 0 | 0.02627 | 0.00675 | 1.084 |
| E10 numeric, seeds 19-21 | 13-26 | 4-8 | 0-5 | 0.0334-0.0356 | | 1.373-1.495 |
| E10 spatial, seeds 19-21 | 14-17 | 9-15 | 0 | 0.0295-0.0318 | | 1.290-1.347 |
| Simulator, sampled hidden physics | | | 0 | 0.02603 | 0 | 1.002 |
| Physics baseline, 0.15 px | 0 | 0 | 0 | 0.02650 | 0.00155 | 1.012 |
| Physics baseline, 0.5 px | 0 | 0 | 0 | 0.02662 | 0.00291 | 1.029 |
| Oracle, first reference | | | 0 | 0.00447 | 0.02440 | 1.435 |

The E10 models also drift in 3-5 of 80 passive samples; no other arm does. Their
same-world ADE is not interpreted (see above).

`projection_holds` passes for all ten checkpoints. Every projected run has zero
penetrating and zero boundary clips. Selected ADE changes by -0.00010 to +0.00030 and
energy score by -0.004 to +0.014 px, all inside the margins. For four of the six E10
models the paired interval of the energy change lies entirely above zero (largest upper
bound 0.027 px), so the cost there is small but measurable.

`learned_geometry_holds`:

- `rest_x16` against `rest_x1`: fails. Penetrating clips fall from 9 to 5 and boundary
  clips from 22 to 8, but the same-world ADE interval is -0.00023 to 0.00065, with the
  upper bound above 0.0005. Energy score rises by 0.017 px.
- `direct_dyn_x1024` against `direct_x0`: holds. Penetrating clips fall from 3 to 0,
  boundary clips from 15 to 14, zero-force drift from 5 samples to 0, and selected ADE by
  0.00074. The same-world ADE interval is -0.00084 to -0.00040, entirely below zero.
  Energy score falls by 0.005 px. RGB contact F1, a weak proxy (D028), moves from 0.320
  to 0.301.

`metric_check` passes: the oracle is worse than the simulator, and no generator is
significantly better.

### Reading

- Scaling the D020 geometry losses buys fewer violations at a cost in motion accuracy.
  The one scale that passed selection did not hold up on the fresh corpus.
- Scaled by 1024, the D029 contact-dynamics losses pass the pre-registered fresh
  test, where D019, D022 and D029 found no learned geometry gain. In this one training
  seed they remove penetration without a loss of accuracy and also remove the direct
  model's zero-force drift. They do not fix canvas clipping (15 to 14 boundary clips),
  so the projection is still needed.
- `direct_dyn_x1024` is not compared with D020 by any registered rule. Against `rest_x1` it
  has lower selected and same-world ADE but a higher energy score (1.084 against 1.055).
- The projection works on every checkpoint tested, including E10 models that never see a
  world, and costs at most 0.014 px of energy score.

## Limits

- One training seed per arm and four validation scenes for selection. The
  `direct_dyn_x1024` result rests on one seed; repeating it with seeds 20 and 21 would
  show whether it is stable.
- Ten fresh ID scene groups; no OOD claim.
- On the fresh corpus every learned checkpoint still has boundary-clipped clips without
  the projection.
- The projection remains an explicit geometric layer, not learned physics.

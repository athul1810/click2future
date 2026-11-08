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

Results: pending.

## Limits

- One training seed per arm and four validation scenes for selection.
- Ten fresh ID scene groups; no OOD claim.
- The projection remains an explicit geometric layer, not learned physics.

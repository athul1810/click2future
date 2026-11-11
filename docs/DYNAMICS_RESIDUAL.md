# D037-D038: force/mass residual and momentum correction

The trajectory prior already receives the visible scene, selected object, force and a
sampled hidden world. Its ordinary pose loss does not explicitly require the generated
acceleration to agree with that force and sampled mass. This study adds two opt-in pieces:

- `dynamics_residual_weight` penalizes finite-difference acceleration error during
  training. Away from object contact and walls, the selected object is matched to
  `force / mass` during the 0.25-second action and the passive object is matched to zero
  external acceleration. Simulator contact and wall masks remove intervals where an
  unknown impulse makes that target incomplete.
- `--momentum-block` corrects pair momentum after the force ends. Before predicted object
  contact, only the selected object's velocity is corrected. After contact, a shared
  centre-of-mass correction preserves the model's relative collision response. The block
  leaves rotation unchanged and skips trajectories approaching a wall, where wall impulse
  is unknown.

The inference order is model sample, momentum correction, then the accepted D028 contact
projection. All additions are optional; a zero residual weight and absent command-line
flags reproduce the previous path.

## D037 development screen

D037 was an unrecorded local development screen that motivated the registered D038
comparison. It has no compact evidence record, and its rule result is not retained or
used as evidence here.

## D038 registered comparison

The rule and seed were committed locally at `9daf9db` before generating the evaluation
corpus. That commit and the result commit first reached GitHub together, so the remote
history does not independently demonstrate the ordering; future registered studies must
push the rule before generating data. Two seed-20 models used the same E7 teacher, cache,
initialization, private RNG and 8,000 updates. Their resolved configs differ only in
`dynamics_residual_weight` (0 versus 4). The fresh seed-980000 corpus contains ten ID
scene groups and 60 conditions, with four generated futures per condition. All 355 stored
simulator clips replay exactly.

The primary comparison is the projected control against the residual-trained model with
momentum correction and projection. It passes all seven comparison checks. The eighth
recorded boolean, post-force momentum conservation, is an implementation check that
passes by construction because the block enforces it.

| Fresh D038 ID | Control + projection | Residual + momentum + projection |
|---|---:|---:|
| Complete generated clips | 240/240 | 240/240 |
| Penetrating / boundary clips | 0 / 0 | 0 / 0 |
| Zero-force / passive drift | 0 / 0 | 0 / 0 |
| Same-world selected ADE | 0.00699 | 0.00582 |
| Direct first-reference pose ADE | 0.02771 | 0.02665 |
| Free-motion dynamics residual | 0.00564 | 0.00101 |
| Post-force momentum error (implementation check) | 0.00865 | 1.64e-8 |
| Multi-reference energy score (px) | 1.2605 | 1.2539 |
| RGB tracker contact F1 | 0.228 | 0.213 |

The scene-paired 95% interval for candidate-minus-control same-world ADE is
`[-0.00149, -0.00084]`; for the dynamics residual it is
`[-0.00595, -0.00328]`. Same-world ADE improves 16.8% overall, while direct
first-reference pose ADE improves 3.8% (`0.02771` to `0.02665`). The same-world score uses
the same sampled masses supplied to the block, so the component breakdown is essential:

| D038 component comparison | Same-world ADE | Paired interval |
|---|---:|---:|
| Projected control | 0.00699 | -- |
| Add momentum block | 0.00618 | [-0.00115, -0.00046] |
| Residual without block | 0.00688 | [-0.00018, -0.00005] |
| Residual added on top of block | 0.00582 | [-0.00046, -0.00028] |

The momentum block accounts for about two thirds of the total same-world reduction. The
wall gain is entirely attributable to the block: wall ADE changes from 0.01887 to 0.01466
with the block, while the residual alone gives 0.01946. The block alone worsens collision
ADE from 0.00773 to 0.00886; the residual brings it back to 0.00782. Without the block,
the residual lowers its trained dynamics error by 80.7% (`0.00564` to `0.00109`) and
raises contact F1 from 0.228 to 0.278.

The supported claim is therefore narrow: on this fresh ID study, the complete opt-in
pipeline improves aggregate matched-world motion while retaining complete tracking,
stationarity and visible geometry constraints. Most of the aggregate accuracy gain comes
from the explicit momentum block; the learned residual's clearest independent result is
lower free-motion dynamics error. It does not show a collision-specific improvement.
Full-pipeline contact F1 falls slightly; it was reported but not gated because simulator
controls already show that this RGB tracker metric is a weak contact proxy. There is one
confirmatory training seed, no OOD claim, and the momentum/projection stages remain
explicit hybrid corrections rather than learned physics.

The compact evidence record is
`docs/results/dynamics-residual-d038.json`. Checkpoints and generated datasets remain
external run artifacts.

## Commands

Train the residual arm with
`configs/training/prior_rest_anchored_dynamics_residual_seed20.json`. At sampling time,
world-conditioned checkpoints accept `--momentum-block --contact-project`; omitting both
keeps raw model output.

## D039 exploratory collision-impulse screen

`collision_impulse_weight` optionally emphasizes the mass-weighted velocity jump at
isolated object contacts. Its target is the clean simulator trajectory, so sampled
friction and restitution are retained rather than replaced by a frictionless collision
formula. The mask excludes wall contact and intervals whose two clean object impulses are
not equal-and-opposite within 5%, which removes external-force contamination. Weight zero
keeps the D038 objective unchanged.

A short matched screen used seed 22, 2,000 updates and the already-opened D038 corpus.
Both arms retained the D038 dynamics residual and used momentum correction followed by
projection at inference; only the collision weight differed (0 versus 1024). The candidate
tracked all 240 clips with zero penetration, boundary violations or drift.

| Opened D038 ID screen | Control | Collision residual |
|---|---:|---:|
| Overall same-world ADE | 0.00565 | 0.00528 |
| Collision same-world ADE | 0.00683 | 0.00606 |
| Direct first-reference pose ADE | 0.02697 | 0.02712 |
| Energy score (px) | 1.2662 | 1.2548 |
| RGB tracker contact F1 | 0.168 | 0.238 |

Collision ADE improves 11.3%, with a scene-paired interval of
`[-0.00117, -0.00043]`. Push ADE improves 7.9%, wall ADE improves 1.5%, and zero-force
output is unchanged. Raw neural collision ADE is 3.2% worse, however; the gain appears
only when both arms use the intended momentum-plus-projection pipeline.

This is enough to retain the loss as an experimental opt-in term, but not to claim a
confirmed improvement: the run is shortened, uses one seed and reuses an opened corpus.
A full matched run on newly registered scenes would be required before promotion. Record:
`docs/results/collision-impulse-screen-d039.json`.

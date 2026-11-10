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

Seed-19 residual and matched-control models were trained locally for 8,000 updates and
scored on a new seed-970000 ID corpus. The residual error fell about 79%, including push,
collision and wall strata. With contact projection, both arms had zero penetration and
boundary violations. A frozen-pose follow-up showed that momentum correction alone could
increase boundary clipping, while momentum followed by projection lowered same-world ADE.
Those observations motivated D038; they are development evidence, not the confirmatory
result.

## D038 registered comparison

The rule and seed were committed at `9daf9db` before generating the evaluation corpus.
Two seed-20 models used the same E7 teacher, cache, initialization, private RNG and 8,000
updates. Their resolved configs differ only in `dynamics_residual_weight` (0 versus 4).
The fresh seed-980000 corpus contains ten ID scene groups and 60 conditions, with four
generated futures per condition. All 355 stored simulator clips replay exactly.

The primary comparison is the projected control against the residual-trained model with
momentum correction and projection. It passes all eight registered checks.

| Fresh D038 ID | Control + projection | Residual + momentum + projection |
|---|---:|---:|
| Complete generated clips | 240/240 | 240/240 |
| Penetrating / boundary clips | 0 / 0 | 0 / 0 |
| Zero-force / passive drift | 0 / 0 | 0 / 0 |
| Same-world selected ADE | 0.00699 | 0.00582 |
| Direct first-reference pose ADE | 0.02771 | 0.02665 |
| Free-motion dynamics residual | 0.00564 | 0.00101 |
| Post-force momentum error | 0.00865 | 1.64e-8 |
| Multi-reference energy score (px) | 1.2605 | 1.2539 |
| RGB tracker contact F1 | 0.228 | 0.213 |

The scene-paired 95% interval for candidate-minus-control same-world ADE is
`[-0.00149, -0.00084]`; for the dynamics residual it is
`[-0.00595, -0.00328]`. Same-world ADE improves 16.8% overall. By grouped action family it
is unchanged for zero force, 40.3% lower for plain pushes, 22.0% lower for wall pushes and
1.17% higher for object collisions.

The supported claim is therefore narrow: on this fresh ID study, the complete opt-in
pipeline improves aggregate matched-world motion while retaining complete tracking,
stationarity and visible geometry constraints. It does not show a collision-specific
improvement. Contact F1 also falls slightly; it was reported but not gated because the
simulator controls already show that this RGB tracker metric is a weak contact proxy.
There is one confirmatory training seed, no OOD claim, and the momentum/projection stages
remain explicit hybrid corrections rather than learned physics.

The compact evidence record is
`docs/results/dynamics-residual-d038.json`. Checkpoints and generated datasets remain
external run artifacts.

## Commands

Train the residual arm with
`configs/training/prior_rest_anchored_dynamics_residual_seed20.json`. At sampling time,
world-conditioned checkpoints accept `--momentum-block --contact-project`; omitting both
keeps raw model output.

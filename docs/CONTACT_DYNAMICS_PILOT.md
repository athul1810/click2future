# D022: local contact dynamics pilot

All 23 D021 neural penetration failures occur in the two directed collision
families: 14/40 and 9/40 clips. Forward, reverse, wall and rest families each
have 0/40 penetration failures. D021 is already opened. These observations
motivate a contact hypothesis; they cannot select or confirm the new candidate.

## Alternatives and choice

- Keep the observable physics baseline as the measured reliability control.
  Its 0/240 D021 penetration failures do not prove exact timing or OOD reliability.
- More whole-clip loss weighting has already failed in D019/D020. Do not repeat
  it without a different, testable mechanism.
- [Graph network simulators](https://proceedings.mlr.press/v205/allen23a/allen23a.pdf)
  predict finite-difference accelerations and integrate them autoregressively.
  This is a feasible representation change for the existing training cache.
- [RiCo](https://arxiv.org/html/2610.12333v1) supports reasoning with current
  motion and local surface neighborhoods. Its full 3D surface/anchor model is
  not a drop-in replacement for our resting single-image interface.
- [IMPACT](https://arxiv.org/html/2611.01372v1) supports explicit free flight
  and bounded learned collision parameters. Its sphere-plane setting does not
  establish superiority for tabletop pair contacts.
- Large force-conditioned video models require separate input/compute adaptation
  and are not the smallest experiment that tests the observed failure mechanism.

Choose a small contact-aware state-transition candidate. This is a justified
experiment, not a claim that a paper or physical law proves it is best.

## Candidate contract

Inputs are frame-zero RGB/mask-derived features and convex hulls, force and one
sampled mass/friction/restitution world. There is no inference-time simulator,
reference recipe, hidden label, future image or post-hoc pose projection.

At 8 fps, maintain relative position/angle and previous finite displacement.
An exact native-240-Hz Newton finite difference supplies force-only motion.
Two shared MLPs predict pair and wall response increments from current motion,
transformed observed hulls, physical latents and local force. SAT distances
restrict pair messages to a reachable neighborhood. This defines locality, not
a hard collision resolution. The allowance is one native pixel, representing
observational quantization, plus the current translational/rotational reach.

Antisymmetric mass-weighted pair increments preserve total finite-difference
linear momentum when walls are out of range. Homogeneous motion scaling makes
rest exact for any weights. These properties do not guarantee energy
dissipation, angular momentum balance, nonpenetration or correct contact timing.
Autoregressive error accumulation remains a substantial risk.

A full clip uses 31 recurrent updates, each with two ordered pair-network calls
and one wall-network call. Report 93 neural module evaluations, rather than
describing this as one-pass video generation.

## Frozen pilot and acceptance

Use the same verified 19,328-example training cache and original validation.
Train from fresh weights, seed 19, 8,000 fixed updates, batch 128 and four-step
autonomous training windows. This is a combined architecture/training candidate,
not a causal isolation of one component from the differently initialized D020.
Use equal native-pixel squared position and finite-displacement errors, with
angular error converted to surface displacement using observed radius.
Sampling strata have mass 0.25 rest, 0.5 contacts and 0.25 other actions.
No claim is made that these settings are optimal.

One TC2 A40, 4 CPUs, 24 GB RAM, at most a one-hour job. Checkpoint optimizer,
EMA, private RNG and source/cache identity every 250 updates. Verify exact CUDA
continuation before the final run. Use detached idempotent Slurm submission;
VPN loss does not resubmit an acknowledged job. Incomplete training is retained
and requires explicit resume. Preserve all attempts and errors.

Before promotion, require fewer than D020's 4/96 validation penetrating clips,
all 96 complete tracks, no rest/passive drift, and selected ADE no worse than
0.0250208912. Retain contact timing and boundary failures as blockers; the
original strict gate stays unchanged. A one-seed validation pass would justify
further experiments, not a final learned-physics success claim. Compare multiple
training seeds and a separately frozen fresh corpus before any confirmatory or
OOD claim. Do not score or tune this candidate on D021 or historical test sets.

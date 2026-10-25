# shared-world relational diffusion

## scope

this is an exploratory follow-up after the team opened the D014 test splits in
job 46716. it does not replace those results, repeat their confirmatory claim or
use test arrays for training/tuning. the present experiment uses the retained
training corpus and the original four validation scenes only. any later test
result must disclose this chronology; a new independent holdout would be needed
for a new confirmatory claim.

E7 adds two changes to the E6 wall backbone:

- local or relational residual adapters after each object/time attention block.
  the relational arm sees the other object's token, noise-scaled relative
  displacement/velocity and observed bounding-box gaps. this is learned
  spatio-temporal denoising, not a recurrent contact solver or hard constraint.
- a per-object mass/friction/restitution latent. true physical labels supervise
  the conditioned arm during training. inference draws the simulator's declared
  independent uniform prior, keeps that draw fixed throughout Heun sampling, and
  uses the same draw for changed forces or selected masks under the same seed.
  the actual hidden properties of the supplied scene are never read.

this is extra physical supervision, beyond E5/E6 pose labels and beyond the
unsupervised E0 flow CVAE. the prior is known from the simulator specification;
we are not claiming to learn or identify hidden physics from one resting image.
four draws are not calibrated uncertainty merely because they differ.

the decoder remains joint SE(2) motion followed by rigid warping of observed
pixels. no simulator, projection, seed filtering, motion override or reranking
runs at inference. all original margins, tracker tolerances and physical scores
stay unchanged. this is an implementation of sampled-world conditioning and
pair adapters, not the full stateful architecture discussed in the research note.

## predeclared comparison

| arm | residual adapters | physical context |
|---|---|---|
| local | per-token, same parameter count | ignored |
| relational | other-object messages and geometric edges | ignored |
| conditioned | same relational messages | labelled in training; prior samples at inference |

adjacent configs may differ only in `relational` and then `world_condition`.
all three have identical parameter shapes/counts, initial tensors, data order,
noise seeds, batch size, losses, update budget and sampler. disabling relational
input replaces the sender with the receiver and edge features with zero; the
adapter computation/capacity is retained. original attention still mixes objects,
so the local arm is not described as a model without interactions.

all arms inherit the same wall EMA at update 15,500, checksum
`cfc618f637f1afb6ba36cb281e895c096288fbc3cbea962fcfc11830513cbeed`.
the old backbone and the zero-output new adapters initially reproduce it exactly.
optimizer, run step and private RNG reset. this is warm-start refinement, not
an independent scratch repeat.

- retain the 336 original training clips plus 2,080 training actions: 19,328
  augmented examples. validate slot/physical-label identity under all square
  rotations/reflections; require the original five warm-start data identities.
- prepare/check the training and validation cache once, before any arm trains.
  both arrays and geometry/world labels have recorded hashes. final tests are not
  read by this preparation path.
- seed 19, 8,000 updates per arm, batch 64, learning rate 1e-4 with the same cosine
  schedule. risk loss is off. all other weights match the prior wall control.
- evaluate the fixed final update 8,000, not the best physics score or validation
  checkpoint. known-world validation denoising loss is a diagnostic, not a prior
  predictive likelihood or checkpoint-selection rule.
- retain the untouched wall anchor and all three arms on every frozen validation
  condition, all four seeds, selection controls, full-horizon zero/passive rest,
  pre-compositing geometry, clipping, contact timing and uncertainty strata.
  execute strict pose-to-RGB replay and collision depth/duration checks natively.
- for every eligible forward/reverse case, hold denoising noise fixed at all four
  seeds and change only selected-object prior mass from 0.65 to 1.85. other mass
  stays 1.25, friction 0.45, restitution 0.2. count correct, equal and reversed
  endpoint effects along the force. preserve raw poses and passive excursions.
  this exploratory diagnostic does not select checkpoints.
- one GPU, four CPUs, 24 GB host memory, 55-minute allocation cap. each training
  process stops after ten minutes and preserves its latest state; an incomplete
  arm is not silently treated as an equal-budget result.
- no checkpoint promotion based on one seed or tracking alone. assess overlap,
  missing outcomes, action consistency and reference coverage together. mean ADE
  does not replace the existing all-case trajectory gate.

the architecture is not a matched main E0/E2/E3 comparison: representation,
supervision, data and effective examples processed differ. all gains or failures
belong to this explicitly labelled follow-up.

the 19,328 augmented examples still come from only 16 independent training
geometries and four hidden-world variants per geometry. the action supplement
adds no new visible worlds. true physical labels reach the conditioned arm's
world encoder; blind arms leave that encoder unused. equal total parameter counts
therefore do not imply equal active capacity or isolate information alone.

## run

use `submission/world_comparison.sbatch` from immutable committed source. its
environment contract is in `submission/README.md`. do not train on the login
node. the driver refuses final-test splits, changed adjacent recipes and unequal
warm-start provenance. raw attempts and all four evaluated ensembles are retained.

## measured validation outcome

all three arms completed 8,000 updates on an a40 in job 46889, from `bcffd58`.
evaluation failed when the uncertainty-strata guard rejected the E7 method name.
`fb72774` extends that guard without changing scoring arithmetic and adds an
evaluation-only recovery path. job 46910 evaluated the same checkpoints using
fresh plans. the original attempt remains intact; its anchor and local videos
reproduce byte-for-byte for all 24 conditions in recovery.

| measure | wall anchor | local | relational | conditioned |
|---|---:|---:|---:|---:|
| complete all-object clips /96 | 95 | 93 | 93 | 96 |
| penetrating clips /96 | 15 | 14 | 14 | 11 |
| penetrating future frames | 142 | 124 | 121 | 105 |
| maximum penetration, pixels | 2.1762 | 2.2166 | 2.2000 | 2.4480 |
| longest consecutive penetration, frames | 22 | 22 | 22 | 19 |
| incomplete wall clips /16 | 1 | 3 | 3 | 0 |
| timed contact-event f1 | 0.3232 | 0.3299 | 0.2828 | 0.3366 |
| forward selected-object spread | 0.00419 | 0.00427 | 0.00423 | 0.02243 |
| forward reference-to-nearest-sample distance | 0.02545 | 0.02550 | 0.02546 | 0.01076 |
| forward sample-to-nearest-reference distance | 0.01189 | 0.01184 | 0.01184 | 0.00958 |
| failed single-reference trajectory cases /24 | 7 | 7 | 7 | 10 |
| correct early-window reversal pairs /16 | 16 | 16 | 16 | 12 |
| correct full-horizon reversal pairs /16 | 16 | 16 | 16 | 16 |

spread and coverage distances use canvas-width units; reference forward spread
is 0.02580. all four models retain zero full-horizon drift in 16/16 zero-force
and 32/32 eligible passive clips. each retains 16/16 correct early selection
targets and contrasts. the conditioned arm has two stationary forward samples
and two stationary reverse samples in the early window; every one moves by the
full horizon. neither window's result is replaced by the other.

world conditioning improves measured forward spread and reference coverage,
where the relational adapter alone barely changes them. with paired denoising
noise, lighter mass increases the selected endpoint along the force in 32/32
trials over eight eligible cases. both blind arms are exactly unchanged in
32/32 trials. these are repeated samples from four scenes, not 32 independent
scenes or evidence of calibrated probabilities.

the conditioned checkpoint is not promoted. fewer clips penetrate, but its worst
penetration is deeper than either control, and the unchanged trajectory gate
fails. the frozen visual case `g03-scene-00730016-collision_1_to_0-p00` is also its
worst-depth case. all four seeds and reference profiles were inspected, along
with every seed of the saved user push. a single-reference ADE penalty can also
increase when plausible hidden worlds differ; the multireference scores stay
visible alongside the existing gate. this does not establish a general contact
model, an external SOTA result or held-out improvement.

## reproducibility

614 local tests and four subtests pass. lint and changed-source formatting pass;
four inherited global-format failures remain. the 257-file source ZIP passes
inventory/link checks, clean-extraction smoke and 14 extracted regression tests.
all three real E7 demo adapters reproduce their offline CPU videos exactly.

all 1,503 collected file digests verify. native replay is exact for 384/384 clips.
independent Windows replay is also exact for 384/384; collision/event counts match
native results. the largest geometry-depth difference is 2.79e-7 pixels. every
arm's two-update CUDA continuation fixture has identical loss and zero model/EMA
difference; this is a device fixture, not an independent training repeat.

job 46909 failed its checksum preflight after one second because the checksum
file used Windows line endings; it performed no evaluation. recovery's launcher
also misspelled the revision environment variable. raw report fields remain
`unrecorded`; the archive and all 419 committed files in the explicitly named
`fb72774` release were checked during and after the allocation. the source
attestation records this limitation separately rather than rewriting reports.
accounting retains all attempts: 1,434 + 1 + 748 = 2,183 allocated GPU seconds,
below the original 55-minute cap in total. no final-test arrays were read.

the [result record](results/shared-world-validation.json) retains every action
family, failed case, prior draw, mass intervention and checkpoint identity. the
[verification record](results/shared-world-checks.json) records the backed-up
assets and checks. full states, cache, videos, poses, both source revisions and
failed attempts are retained privately.

## next work

1. expand independent training geometries and physical worlds; extra actions on
   the same 16 geometries cannot establish generalization to unseen worlds.
2. test explicit learned contact-state transitions against the present relational
   control, retaining shared worlds and all depth/duration/action diagnostics.
   adding pair attention alone did not solve contact.
3. repeat the useful conditioning comparison with independent training seeds and
   a newly frozen holdout. the opened D014/D015 sets cannot support a new claim
   of confirmatory improvement. keep the required E0/E2/E3 study and demo bundle
   separate from this additional-supervision branch.

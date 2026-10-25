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

## run

use `submission/world_comparison.sbatch` from immutable committed source. its
environment contract is in `submission/README.md`. do not train on the login
node. the driver refuses final-test splits, changed adjacent recipes and unequal
warm-start provenance. raw attempts and all four evaluated ensembles are retained.

measured outcomes will be appended after the allocation; unit checks establish
implementation behaviour, not learned quality.

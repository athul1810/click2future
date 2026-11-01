# local surface message prototype

## scope

the [1 november research review](RESEARCH_2026-11-01.md) identifies a specific
representation gap: E7's detailed observed hulls supervise the training loss,
but its learned messages use coarse object descriptors. this prototype tests
local surface correspondence before object pooling. FIGNet, HOPNet and RiCo
already establish geometry-based contact reasoning in other settings; this is
an adaptation to our restricted single-image task, not a new general principle.

the full model has 1,079,944 trainable parameters: the original E7 backbone plus
15,364 surface-head parameters. no backbone weights are frozen. both arms load
identical E7 EMA tensors and identical new tensors. the final surface head starts
at zero, preserving the teacher's initial predictions.

## observable geometry and learned messages

`models/surface_motion.py` consumes the hull vertices, edge normals and valid
prefix already extracted from the observed masks. the caller does not provide
true geometry, material labels or future contacts at inference. at each denoising
call, the current predicted rigid poses transform those observed hulls.

for each valid vertex, the encoder reads its rotated offset/normal, closest
other-object segment displacement/normal/distance, current point motion and
relative motion, canvas distances, local selected force, own/other sampled
physical world and diffusion noise level. a shared point MLP followed by masked
mean/max pooling predicts an EDM-scaled residual. the same E7 backbone is refined
through the complete gradient path.

the two arms have identical learned parameters:

| arm | cross-object surface features before the shared MLP |
|---|---|
| local | keep the vertex-to-segment correspondence and relative motion |
| pooled | broadcast the mean correspondence and relative motion over valid vertices |

the control retains own-surface details and boundary distances. this tests local
correspondence against pooled interaction information, not geometry against no
geometry. canonical spatial slots and shared prior seeds remain unchanged.

closest-point coordinates are bounded to each segment for geometric feature
calculation. **generated poses are not projected or clamped.** there is no
contact solver, output reranking, contact gate, analytic dynamics integration or
special rest override. geometric features can inform the learned model without
guaranteeing valid physics. a failure remains possible and must be measured.

## implementation and verification

- `training/train_surface.py` reuses the existing contact objective and training
  strata with an immutable world/hull binding. it verifies original teacher data
  identities, allows only the explicit checked coverage extension, and resets
  optimizer/private RNG. format 5 checkpoints retain full model, EMA, optimizer,
  RNG, configuration, source hashes and data provenance.
- `training/surface_checkpoint.py` provides strict isolated loading and observable
  RGB sampling. historical E5-E8 model files, public loaders and evaluation
  sources are untouched. format 5 is intentionally not registered as an official
  evaluated method yet; it must not be labelled E7 or inherit E7 scores.
- unit checks cover analytic segment distance/gradient, object permutation,
  padding isolation, degenerate rotations, matched initial tensors/parameters,
  exact teacher migration, full backbone/head gradients, strict load/resume and
  repeatable observable-only sampling. CPU optimizer steps are tiny synthetic
  test fixtures; real training CLI requires scheduled TC2 CUDA.
- on the real saved push, both zero-output arms reproduce all four teacher clips
  pixel-for-pixel. measured CPU cost was about 0.8 seconds for four clips with
  four threads on this machine. this is an initial compatibility/cost check,
  **not a trained-quality result or a GPU latency claim**.

## training gate

retrieve #68 job 47305 before freezing a new cache/teacher and matched training
recipes. the outcome may change whether broader data is useful. the current
prototype does not contain a selected learned checkpoint and is not a default
demo replacement.

before any new run: freeze equal batches, losses, initial tensors, private RNG,
sampling weights, seed 19, update count and final-update selection. exercise exact
continuation on the allocated GPU first. train both arms; retain incomplete and
failed attempts. add a fresh, explicitly labelled format-5 evaluation path and
freeze its source before sampling. do not alter old plans or relabel the model
to bypass checkpoint guards.

judge all declared validation seeds, completeness, depth/duration, contact timing,
rest, action horizons, full selection drift, force scaling and multireference
coverage together. independent training repeats and a newly frozen holdout are
still necessary for a confirmatory improvement. the single-image hidden-world
ambiguity and limitations of the 64px rigid decoder remain.

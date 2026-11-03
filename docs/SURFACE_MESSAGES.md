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
  RGB sampling. the initial prototype left public dispatch unchanged. the
  2 november registration below adds a distinct E9 path; it cannot inherit E7 scores.
- unit checks cover analytic segment distance/gradient, object permutation,
  padding isolation, degenerate rotations, matched initial tensors/parameters,
  exact teacher migration, full backbone/head gradients, strict load/resume and
  repeatable observable-only sampling. CPU optimizer steps are tiny synthetic
  test fixtures; real training CLI requires scheduled TC2 CUDA.
- on the real saved push, both zero-output arms reproduce all four teacher clips
  pixel-for-pixel. measured CPU cost was about 0.8 seconds for four clips with
  four threads on this machine. this is an initial compatibility/cost check,
  **not a trained-quality result or a GPU latency claim**.

the complete stack passes 644 tests plus four subtests, with six inherited
torchvision warnings. lint and changed-source formatting pass; four inherited
global-format failures remain. the 299-file source ZIP passes inspection,
clean-extraction smoke and ten extracted regression tests. source `9b63e0f`
contains the tested prototype; later changes only record this evidence.

[verification record](results/surface-prototype-checks.json) retains identities,
runtime and the current TC2 access limitation. the
[private draft backup](https://github.com/athul1810/click2future/releases/tag/untagged-60da86aad5f716d7aa68)
contains both immutable sources, raw action-probe videos and references,
all-seed failure sheets, initial compatibility output and full test logs. all
five release assets were downloaded and verified. this backs up implementation
and development diagnostics, not a trained surface-model result.

## training gate

job 47305 has been collected and reviewed in #68. neither coverage arm is
promoted. the new comparison therefore uses the original 19,328-example training
cache and original conditioned E7 teacher, rather than either expanded cache.
the trained comparison is now collected below. neither arm is promoted.

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

## 2 november run freeze

`surface_local.json` and `surface_pooled.json` differ only in `local_surfaces`.
both use seed 19, 8,000 updates, batch 64, the same existing contact objective,
AdamW learning rate 1e-4, all 1,079,944 parameters trainable, and the fixed final
checkpoint. each arm has a 1,200-second training cap. a partial run is retained
and fails the comparison instead of being presented as a completed budget.

the real-cache initialization check verifies identical initial tensors, private
RNG and sampling weights. both zero-output heads preserve all four teacher clips
pixel-for-pixel on the first sorted validation push. this remains an initialization
check, not a training result.

the public sampler and fresh evaluation plans now register format 5 as **E9**.
observable hulls enter only that sampler path. the generic suite, geometry audit,
stratification and world probes accept the distinct method without changing metric
formulas, thresholds or historic plans/reports. new plans bind this evaluator
revision; old official plans must still be replayed with their pinned source.
the isolated historical audit verifies their saved source identities as before.
the checkpoint adapter exposes the surface variant and its inference source hashes.

the integration test samples, scores and pixel-replays a tiny format-5 fixture,
and rejects an attempted E7 alias. legacy E5-E8 tests remain in the full suite.
`submission/surface_comparison.sbatch` runs both arms, exact CUDA continuation
fixtures, all validation/selection samples, geometry/contact/rest/uncertainty
audits, shared-world interventions and the complete force/selection response
diagnostic. no old test split is opened.

## completed screen, collected 3 november

job **50260** completed `0:0` on an A40 in 1,819 allocated GPU seconds. both
arms finish all 8,000 updates and both CUDA continuation fixtures are exact.
all 926 retained file hashes pass; all 490 frozen source files verify before
and after the job. independent Windows pose replay is pixel-exact for all 288
clips and reproduces native event counts. raw failed transport attempts remain
recorded; no optimization ran during those attempts.

| measure | E7 teacher | local | pooled |
|---|---:|---:|---:|
| complete all-object clips /96 | 96 | 96 | 96 |
| penetrating clips /96 | 11 | 12 | 11 |
| penetrating future frames | 105 | 105 | 108 |
| maximum depth, pixels | 2.4480 | 2.5035 | 2.5176 |
| longest penetration, frames | 19 | 19 | 19 |
| timed contact F1 | 0.3366 | 0.3600 | 0.3673 |
| failed trajectory conditions /24 | 10 | 10 | 10 |
| native full-horizon correct targets /64 | 63 | 64 | 64 |
| native full-horizon passive drifts /64 | 1 | 0 | 0 |
| native full-horizon force comparisons complete /76 | 76 | 75 | 76 |
| native completed force comparisons increasing | 76 | 75 | 76 |

local correspondence does not outperform pooled interaction information overall.
both refinements slightly improve some response measures and timed contact F1,
but worsen maximum depth; local adds a penetrating clip and loses one complete
force comparison. neither passes overall quality. the small ADE differences
do not resolve the ten failing trajectory conditions. one seed and four reused
validation scenes cannot establish a general result about local contact models.

all response rows here come from the same A40 allocation. eight selection
conditions and thirteen force conditions are excluded at full horizon because
their references contact first. missing predictions remain in the denominators.
the teacher's 96 primary clips are byte-identical to job 47305, but its
supplementary native response is 63/64 selection and 76/76 force completeness,
versus 62/64 and 75/76 in the earlier Windows CPU probe. this is a platform and
threshold boundary, not a learned improvement. historical CPU records are kept.

the [complete result](results/surface-validation.json) retains every action
family, failed case, depth/duration, stationarity, uncertainty, response and
mass-intervention result. E9 remains an experimental comparison; the E7 teacher
is not replaced on this evidence.

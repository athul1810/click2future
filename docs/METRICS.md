# Evaluation implementation

T07 includes trajectory/collision scoring, RGB object tracking, and an experimental
contact estimator. Tracking works on the measured solid-colour simulator clips. Contact
event detection is currently too inaccurate to support model-comparison claims.

## Controlled contact stress check

Candidate dataset-0.2 from PR #31 exposes a distinction between collision occurrence
and event timing. On its initial held-out check, timed-event F1 was 0.321 even though
23 of 24 colliding clips had at least one detected encounter, with no false collision
clips. This motivated the separate encounter diagnostic; it did not change the tracker
or contact thresholds.

A fresh dataset seed, 193000, then checked the diagnostic on 63 clips from three new
held-out scene groups. All 3,906 future object observations were tracked. Encounter
counts were TP=23, FP=0, FN=1, TN=39: F1=0.979 and accuracy=0.984 at positive prevalence
24/63. Timed-event F1 remained 0.551. These are extractor checks on simulator clips,
not generated-model results. Three scene groups and existing solid-colour shape
families do not establish general reliability. Keep timed-event errors visible.

The recorded data revision, original check and fresh check are in
`results/controlled-extractor-checks.json`. The data PR still needs its model force-range
integration fixed before training uses it.

## Input contract

Use fixed object identities established from frame 0. Do not reorder objects or choose
different generated samples to improve individual frame errors. Coordinates are normalized
image coordinates, with x right and y down. Keep padding in the same slots across arrays.

The command reads an NPZ archive with these fields:

| Field | Shape | Type |
|---|---|---|
| `predicted_positions` | `[K,T,N,2]` | Numeric |
| `reference_positions` | `[T,N,2]` | Numeric |
| `object_valid` | `[N]` | Boolean |
| `tracked` | `[K,T,N]` | Boolean |
| `predicted_collision_onset` | `[K,T-1,N,N]` | Boolean |
| `reference_collision_onset` | `[T-1,N,N]` | Boolean |
| `collision_observed` | `[K,T-1,N,N]` | Boolean |
| `timestamps` | `[T]` | Finite seconds, increasing from zero |

T includes frame 0. It is excluded from trajectory errors. Collision interval 0 is the
transition from frame 0 to frame 1. Collision matrices must be symmetric with a false
diagonal for valid objects. Supply contact onsets, not persistent contact states or raw
probabilities. Apply any classifier threshold using a validation-selected rule upstream.

A tracker adapter must mark failed observations in `tracked` and unreliable collision
bins in `collision_observed`. Do not replace missing detections with stationary objects
or negative collision labels. NaNs are allowed only for untracked or padded positions.
Predicted coordinates outside [0,1] remain scored and are counted separately; reference
coordinates must be valid. Wall contacts are excluded from this object-pair interface.

## Current scoring policy

- ADE averages Euclidean distances across future frames and valid objects. FDE averages
  distances at the final future frame. Both all-object and selected-object scores are returned.
- A sample with any missing future observation has no ADE/FDE for that object set.
  Means are explicitly labelled as averages over complete samples. Counts, observation
  denominators, and missing-track rates accompany them. Do not report these means alone.
- Best-of-K is available only when all K samples are complete. Minimum ADE chooses one
  entire sample. Minimum FDE is chosen separately; FDE at the minimum-ADE sample is also
  returned. Ties use the earliest sample index.
- Interval collision scores count each valid unordered object pair once and apply no
  temporal tolerance. Reports include confusion counts, accuracy, precision, recall,
  F1, positive prevalence, and known/missing bin counts.
- Event scores use one-to-one matching for each pair within the explicitly supplied
  interval tolerance. Sorted events are matched earliest first. A prediction can match
  at most one reference. A pair with any missing interval is excluded from event scoring,
  with complete and expected pair-timeline counts reported.
- Encounter scores classify whether each valid unordered pair has any contact onset
  during the full clip. They require a complete observed pair timeline and report
  confusion counts, accuracy, prevalence, precision, recall and F1. They measure
  collision occurrence; they do not measure timing, repeated events or wall contacts.
- Undefined ratios are JSON `null`. If neither reference nor prediction has an event,
  event precision, recall, and F1 are null. If predictions exist without reference events,
  precision and F1 are zero and recall is null. A single valid object has no pair bins.

The extracted-array scoring policy is metric version 1. RGB reports use version 2,
which respects optional `frame_valid` masks: missing RGB removes the corresponding
tracks and contact observations, even when placeholder pixels look valid. Do not mix
old and new RGB reports in a comparison. Review against simulator fixtures
and freeze the tolerance and extraction rules on validation data before model comparisons.
Collision reports additionally use `collision_metric_version: 2` when encounter
scores are present. Legacy collision tables remain readable, and mixed versions are
rejected. Keep encounter and timed-event results together when interpreting contact behavior.
Per-sample records preserve counts for [scene-group aggregation and comparison
tables](EVALUATION_REPORTS.md), including missing observations and bootstrap intervals.

## Run

From the configured development environment, after preparing an extracted input archive:

```bash
python -m click2future.evaluation \
  --input outputs/extracted-condition.npz \
  --output outputs/metrics-condition.json \
  --source video_tracking --extractor-version tracker-v1 \
  --run-id E2-validation-seed0 --sample-id reference-0001 \
  --scene-group-id scene-0001 --condition-id condition-0001 \
  --selected-object 0 --event-tolerance 1
```

The example tolerance is one recorded-frame interval, not one second; it is not a frozen
experiment setting. Replace the IDs and extractor version with the actual run metadata.

Use `--source auxiliary_head` for head predictions and `--source control_fixture` for
synthetic test cases. These labels remain in the output, so head scores cannot silently
be presented as generated-video scores. Video tracking requires an extractor version.
Every successful report includes the input archive SHA-256, identifiers, timestamps,
metric version, and denominators. Corrupt inputs produce a failed record and exit code 1.
A passed record means metric computation succeeded, not that model quality is good.

Run the known controls without a simulator, GPU, or model checkpoint:

```bash
python -m pytest tests/test_evaluation_metrics.py tests/test_evaluation_cli.py -q
```

The controls cover exact, shifted, static, and missing trajectories; whole-sample
selection; padding; event timing and duplicate matches; empty event sets; corrupt inputs;
and source-labelled JSON export. These are test fixtures, not project results.

## Extract from RGB videos

Save generated clips under `videos` in an NPZ as uint8 `[K,T,H,W,3]`, including the
unchanged first frame. The reference is a simulator JSON/NPZ pair with matching dimensions.
From the configured environment:

```bash
python -m click2future.evaluation.video \
  --reference-dir outputs/pilot_32f --sample-id scene-00000032-variant-001 \
  --videos outputs/generated.npz --config configs/evaluation/pixel.yaml \
  --run-id E2-validation-seed0 --output outputs/video-metrics.json \
  --extracted-output outputs/extracted.npz
```

The tracker uses RGB pixels and shared reference frame-0 identity masks. Those masks
initialize evaluation identities only and are not supplied as additional model inputs.
Future oracle masks, trajectories, collisions, and hidden physics are excluded from
extraction. All compared models must use the same initialization.

Objects are matched to their frame-0 colours and separated into connected components.
Similar colours, disappearance, fragmentation, and large area changes produce missing
observations. Partial occlusion can still bias accepted centres. This baseline supports
solid colours; textures and generated-video artefacts need separate validation.

The current contact estimator combines pixel proximity with opposing velocity changes
projected along the line between object centres. It fits short trajectory windows in
pixel/frame units, then selects separated peaks. `motion_rule: opposing` retains the
earlier vector-dot-product rule. All observations in the motion window must be tracked.
This is offline evaluation:
it uses future RGB context, but no future oracle labels. The initial velocity is assumed
zero under the simulator contract. The returned `state` is pixel proximity, not confirmed
physical contact. Close repeated events, small impulses, and wall interactions remain hard.

## Simulator extraction measurements

The current [normal-motion record](results/contact-normal-calibration.json) uses validation
scene groups 20000-20099 and fresh held-out groups 120000-120099, two physics variants
each. Do not put these reserved groups in model datasets. Reproduce it with:

```sh
python -m click2future.evaluation.calibrate --method normal_motion --held-out-start 120000 --output outputs/contact-normal.json --config-output outputs/pixel-normal.yaml
```

The validation search used windows 3/5/7, thresholds 0.04/0.08/0.12/0.2/0.3/0.4, and
gaps 0/1 pixel. Maximum validation F1 selected window 5, threshold 0.08, and gap zero
(validation F1 0.717). Settings were frozen before generating the new held-out set.
The active `configs/evaluation/pixel.yaml` uses this rule.

| Same 200 fresh held-out clips | Previous motion rule | Line-of-centres rule |
|---|---:|---:|
| Reference events | 22 | 22 |
| Matched events | 4 | 13 |
| Extra events | 5 | 8 |
| Missed events | 18 | 9 |
| Event precision | 0.444 | 0.619 |
| Event recall | 0.182 | 0.591 |
| Event F1 | 0.258 | 0.605 |

All 18,538 future object observations and 662 pair timelines were observed. Mean
centroid error was 0.218 pixels. Only 22 true events were present; these are preliminary
clean-simulator estimates, not model-quality results or calibrated event probabilities.
The centre line approximates the contact normal and can be wrong for rectangles and
glancing collisions. Rasterization, small impulses, wall interactions, and closely
spaced events still cause errors. Read collision scores with these failure counts.

The separate [gap search](results/contact-gap-calibration.json) did not beat the old
setting on validation. It used held-out groups 110000-110099, which were not reused for
the normal-rule comparison. The expanded modes reject previously inspected held-out
ranges. Further rule changes require fresh held-out groups.

The earlier [motion record](results/contact-motion-calibration.json) is retained:

```bash
python -m click2future.evaluation.calibrate \
  --output outputs/contact-motion-calibration.json \
  --config-output outputs/pixel-calibration.yaml
```

There are 100 validation scene groups beginning at visible seed 20000 and 100 fresh held-out
groups beginning at 40000. Each has two hidden-physics variants, 32 frames, 64px and 8 FPS.
Reserve these groups from model datasets. The earlier 30000-30029 held-out groups were not
used for this iteration. Future changes require another fresh held-out set.

Validation tested motion windows of 3 or 5 frames and thresholds 0.1, 0.2, 0.3, or 0.4
pixels/frame, holding the proximity gap at zero and event tolerance at one interval.
Maximum validation event F1 selected window 3 and threshold 0.3 (validation F1 0.50).
These settings were fixed before the new held-out evaluation. The table compares both
estimators on exactly the same fresh clips.

| Held-out contact measurement | Pixel proximity | Proximity plus motion |
|---|---:|---:|
| Reference events | 30 | 30 |
| Matched events | 4 | 7 |
| Extra events | 38 | 8 |
| Missed events | 26 | 23 |
| Event precision | 0.095 | 0.467 |
| Event recall | 0.133 | 0.233 |
| Event F1 | 0.111 | 0.311 |

All 18,538 future object observations were tracked across 200 clips, with mean centre
error 0.235 pixels and 95th percentile 0.459 pixels. All 662 object-pair timelines were
complete. No events were excluded for missing tracks. The detector improves but its low
recall still prevents reliable model-comparison claims. Interval accuracy is dominated
by negative bins. These are clean simulator results, not generated-video results, and
variants within a scene are correlated. No confidence intervals are reported.

The [original proximity record](results/pixel-extractor-calibration.json) is retained.
Its 60-clip scores can be reproduced with `--method proximity --groups-per-split 30
--held-out-start 30000`; do not compare its F1 directly with a different held-out set.

## Action controls

Action checks are implemented in `evaluation.actions` and `evaluation.controls`. The
fixed configuration is [actions.yaml](../configs/evaluation/actions.yaml): the first
0.5 seconds (frame 4 at 8 FPS), with one pixel at 64px as the motion threshold. The
reference controls must start at rest and stay free of object/wall contacts throughout
that window. They must share the same objects, hidden physics, initial image, selected
object and timing, changing only the force. A failing reference control rejects the
whole comparison; it is not silently removed after inspecting model outputs.

- Nonzero force: direction error between force and selected-object displacement.
  Stationary samples get no angle, with stationary and missing counts reported separately.
- Zero force: maximum displacement from frame 0 over all valid objects and all frames
  in the window. An object moving away and returning is still counted as drift.
- Reversed force: pair samples by sampling seed. Report the angle from an opposite
  response and whether both displacements have the correct force signs. Stationary
  pairs remain in the complete-pair denominator; missing pairs are counted separately.

Reproduce the [16-group simulator control check](results/action-control-checks.json):

```bash
python -m click2future.evaluation.controls --simulator-check \
  --output outputs/action-control-checks.json
```

The fixed fixture family uses seeds 50000-50015, eight force directions, both selected
object slots, and matched forward/reversed/zero-force clips. All 16 groups had complete
tracks, correct forward/reversed directions and force signs, and zero measured zero-force
drift. These controlled fixtures validate the code; they do not establish model quality.
Tests also cover perpendicular/wrong-direction motion, frozen output, missing tracks,
out-and-back drift, changed hidden physics, contact-contaminated windows, and invalid inputs.

For model samples, prepare reference JSON/NPZ pairs and a generated-video NPZ containing
`forward`, `reverse`, and `zero`, each uint8 `[K,T,H,W,3]`. Use the same ordered sampling
seeds for all three actions and retain the exact first frame:

```bash
python -m click2future.evaluation.controls \
  --reference-dir outputs/control-references \
  --forward-id forward --reverse-id reverse --zero-id zero \
  --videos outputs/control-samples.npz --sample-seeds 0 1 2 3 \
  --run-id E2-controls --output outputs/action-metrics.json
```

Records distinguish `generated_video` from `simulator_control`, include input hashes and
configuration, and retain sample counts. A `passed` status means calculation succeeded,
not that the model follows the action. The CLI checks seed count/order declarations;
the sampler remains responsible for actually using those seeds.

## Image quality

`evaluation.images` computes frame-aligned SSIM and LPIPS against one reference future.
Both exclude frame 0 and use stored RGB values at native resolution. Inputs are uint8
`videos [K,T,H,W,3]`; floats and mismatched shapes are rejected. The NPZ may also contain
boolean `frame_valid [K,T]` for missing RGB frames. Object-tracker failures do not remove
frames from image evaluation. Each report retains per-frame scores, missing counts, and
per-clip means. A clip with any missing future frame has a null mean; aggregate means
include complete clips only and must be presented with their counts.

Install the optional dependencies alongside the selected Torch backend:

```bash
uv sync --locked --extra cpu --extra image-metrics
```

SSIM uses scikit-image 0.26.0 with `data_range=255`, RGB channel averaging, an 11-pixel
Gaussian window, sigma 1.5, population covariance, and K1/K2 of 0.01/0.03. The library
excludes the five-pixel window border. There is no resizing or colour/gamma conversion.
These settings follow the [scikit-image documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity).

LPIPS uses lpips 0.1.4, AlexNet with ImageNet weights, and learned calibration version 0.1.
Inputs become float32 NCHW in [-1,1], following the [official implementation](https://github.com/richzhang/PerceptualSimilarity).
The network runs frozen in evaluation/inference mode, without resizing. The evaluator
requires height and width of at least 64 pixels. Package versions, weight SHA-256 hashes,
preprocessing, and the selected device are saved in every LPIPS report. TorchVision 0.26.0
matches our Torch 2.11.0 dependency, following the [official version pairing](https://pytorch.org/get-started/previous-versions/).

The first LPIPS run needs the official AlexNet weights in the Torch cache. Allow that
initial download explicitly; later runs work from the cache. Missing or corrupt weights
fail the run rather than fall back to random features. Cache/model files stay outside Git
and the final source-code ZIP.

```bash
python -m click2future.evaluation.images \
  --reference-dir outputs/pilot_32f --sample-id scene-00000032-variant-001 \
  --videos outputs/generated.npz --source generated_video --run-id E2-validation-seed0 \
  --metrics ssim lpips --download-weights --output outputs/image-metrics.json
```

Omit `--download-weights` after caching. Use `--metrics ssim` for SSIM alone; it loads no
LPIPS model. Input hashes and reference/run/scene/condition IDs allow these reports to be
joined with the trajectory/contact reports. `--batch-size` controls LPIPS memory use.
Each requested metric must succeed; failures produce a failed record and exit code 1.
A metric omitted from the request is not reported as a zero score.

Reproduce the checked-in [image-control measurements](results/image-quality-controls.json):

```bash
python -m click2future.evaluation.image_controls \
  --output outputs/image-quality-controls.json
```

Add `--download-weights` only if needed. These checks use four seeded simulator scenes,
16 frames at 64px, and exact copies, four-pixel shifts, Gaussian noise, and frozen-frame
controls. All preserve frame 0. The shift rolls the whole future frame with wraparound;
noise has standard deviation 12 before clipping to uint8. Seeds are 60000-60003.

| Control | Mean SSIM | Mean LPIPS |
|---|---:|---:|
| Exact copy | 1.0000 | 0.0000 |
| Four-pixel shift | 0.7498 | 0.1475 |
| Gaussian noise | 0.4096 | 0.2265 |
| Repeat frame 0 | 0.9517 | 0.0263 |

The table averages each clip's future-frame mean across four scenes. These are metric
checks, not trained-model results. A large static background can give frozen output good
image scores, and a valid alternative physical future may score poorly against one
reference. Report image quality alongside action, trajectory, and diversity measures.

## Temporal consistency and ensemble diversity

`evaluation.futures` tracks each generated video once and computes temporal diagnostics,
whole-trajectory diversity, finite-reference coverage, and contact-event frequencies.
All reference variants must share the same visible scene, first frame, object identities,
action, and timestamps. Only hidden physics and the future may differ. Reference future
positions/contact labels are used for scoring, never for model conditioning or tracking.

```bash
python -m click2future.evaluation.futures \
  --reference-dir outputs/pilot_32f \
  --reference-ids scene-00000032-variant-000 scene-00000032-variant-001 \
  --videos outputs/generated.npz --source generated_video --run-id E2-validation-seed0 \
  --output outputs/future-metrics.json
```

Use NPZ `videos` as uint8 `[K,T,H,W,3]`, optionally with boolean `frame_valid [K,T]`.
Missing RGB frames invalidate their object observations and temporal windows; they are
not interpreted as stationary objects. The report records input hashes, reference IDs,
timestamps, tracker/contact settings, metric versions, and extraction failure counts.
Use the same ordered reference set and settings for every compared model. The pixel
config checks native resolution and frame timing.

The temporal report contains these per-sample diagnostics:

- Object RGB error after compensating for centroid translation between consecutive
  future frames. RGB is sampled bilinearly; all source taps must lie within the object.
  Report normalized absolute error together with scored/expected object transitions,
  candidate/scored pixel counts, and the valid pixel fraction. Rotation and occlusion
  reduce coverage and are not reconstructed by this translation-only method.
- Background RGB change on pixels outside all tracked objects at both endpoints.
  An interval is unscored when any valid object is missing. Counts accompany its mean.
- Trajectory acceleration norm in normalized image units per second squared. It uses
  the actual timestamps and future-frame triples. Windows touching the union of declared
  reference object/wall contacts, with a fixed one-interval margin, are excluded. Applied
  force intervals are excluded for the selected object. The same exclusion windows are
  used for all models, with excluded/missing/scored counts retained.

Frame 0 is excluded from transitions and acceleration triples. Acceleration is a
smoothness diagnostic, not a physical-validity score. Rasterization creates a nonzero
floor. Generated collisions outside the reference windows can still raise it, and a
frozen output can score zero on every temporal diagnostic.

Trajectory diversity is the mean Euclidean distance over complete whole future clips,
reported for all objects and the selected object. It includes endpoint spread, per-pair
records, expected/scored pair counts, and reference-set spread. K=1 has no pairwise spread
and returns null. Duplicate samples have zero spread. Colour flicker alone does not
create trajectory diversity.

Coverage averages the distance from each reference future to its nearest generated
whole trajectory. The reverse direction averages each sample's distance to its nearest
reference. Both require all K sample trajectories to be complete for that object set.
The matching cannot select a different sample for each frame or object. These distances
use a finite reference set: an unmatched generated future can still be physically valid.
Reference positions are oracle values while generated positions have extraction error.

The contact-frequency table reports, for each unordered object pair, the fraction of
futures with any onset and the empirical Bernoulli variance. Every K timeline must be
observed before its sampled rate is reported. Reference frequencies use simulator labels;
sampled frequencies inherit the current contact estimator's low recall. Do not treat
frequency agreement as a reliable physical result yet.

Reproduce the [simulator control record](results/future-metric-controls.json):

```bash
python -m click2future.evaluation.future_controls \
  --output outputs/future-metric-controls.json
```

The checks use four visible scenes, seeds 70000-70003, with eight hidden-physics futures
per scene and four samples per control, at 64px/8 FPS for 32 frames. The physics control
uses the first four reference videos as metric inputs; it is not an independent model
sample. Other controls duplicate one future, repeat frame 0, or alternate +/-8 RGB values
without changing motion. All preserve frame 0. The table averages per-scene scores:

| Control | Trajectory spread | Reference-to-sample distance | Object RGB error |
|---|---:|---:|---:|
| Hidden-physics variants | 0.01442 | 0.00520 | 0.00000 |
| Duplicate future | 0.00000 | 0.01172 | 0.00000 |
| Frozen first frame | 0.00000 | 0.03921 | 0.00000 |
| Colour flicker | 0.00000 | 0.01172 | 0.06275 |

Distances use normalized coordinates; RGB errors are divided by 255. Background flicker
error was also 0.06275. All controls passed. Frozen frames had zero acceleration; the
moving physics controls averaged 0.09605 normalized units/second squared after exclusions.
These cases show why temporal scores, action adherence, image quality, and diversity
must be interpreted together. They validate implementation behaviour, not model quality.

## Remaining work

The implementation and simulator validation are ready for review. Contact extraction
remains approximate, and trained-generator/OOD checks belong with the model experiments.
The untrained CVAE format check retains all 248 missing observations instead of hiding
them. T12 has [a joint objective and head export](MODEL_TRAINING.md), with CPU gradient
checks; the dataset diffusion trainer and sampler remain. The uncertainty experiments
in T16 need actual model samples; these metric controls do not complete that experiment.

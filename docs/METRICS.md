# Evaluation implementation

T07 includes trajectory/collision scoring, RGB object tracking, and an experimental
contact estimator. Tracking works on the measured solid-colour simulator clips. Contact
event detection is currently too inaccurate to support model-comparison claims.

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
- Undefined ratios are JSON `null`. If neither reference nor prediction has an event,
  event precision, recall, and F1 are null. If predictions exist without reference events,
  precision and F1 are zero and recall is null. A single valid object has no pair bins.

These policies are versioned as metric version 1. Review them against simulator fixtures
and freeze the tolerance and extraction rules on validation data before model comparisons.
Per-sample records preserve counts for later aggregation; no dataset-level averaging or
scene-group bootstrap is implemented yet.

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

The current contact estimator combines pixel proximity with opposing changes in object
velocity. It fits short trajectory windows in pixel/frame units, then selects separated
peaks. All observations in the motion window must be tracked. This is offline evaluation:
it uses future RGB context, but no future oracle labels. The initial velocity is assumed
zero under the simulator contract. The returned `state` is pixel proximity, not confirmed
physical contact. Close repeated events, small impulses, and wall interactions remain hard.

## Simulator extraction measurements

Reproduce the current [contact measurement record](results/contact-motion-calibration.json):

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

## Remaining work

Contact recall and checks on generated videos remain unresolved. Image metrics, temporal
consistency, ensemble diversity/coverage, wall-contact scoring, and scene-group aggregation
still need implementation. T07 remains open. The action-control tools are ready to use
when trained samples become available for T16.

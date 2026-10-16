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

The contact estimate measures gaps between visible pixel regions. A new contact requires
both objects to be tracked at both interval endpoints. Persistent contact is not repeated,
and missing frames cannot create an onset on reappearance. Rasterization and contacts
between recorded frames can shift or hide events. Wall events are not extracted.

## Simulator extraction measurements

Reproduce the checked-in [measurement record](results/pixel-extractor-calibration.json):

```bash
python -m click2future.evaluation.calibrate \
  --output outputs/pixel-extractor-calibration.json \
  --config-output outputs/pixel-calibration.yaml
```

The run uses 30 validation scene groups and 30 held-out groups, each with two hidden-physics
variants, at 64px and 32 frames. Validation visible seeds start at 20000; held-out seeds
start at 30000. Reserve these groups from model datasets. Gap candidates are 0, 0.5, 1,
and 1.5 pixels. Validation event F1 selected 0 pixels; event tolerance was fixed at one
recorded interval before evaluation. Other tracker settings were fixed heuristics.

| Held-out measurement | Result |
|---|---:|
| Clips | 60 |
| Tracked future object observations | 5580 / 5580 |
| Mean centre error over tracked observations | 0.249 pixels |
| 95th percentile centre error | 0.500 pixels |
| Complete object-pair timelines | 200 / 200 |
| Reference collision onsets | 19 |
| Matched / extra / missed contact events | 2 / 17 / 17 |
| Contact event precision / recall / F1 | 0.105 / 0.105 / 0.105 |

These are clean simulator extraction results, not model results. The 99.4% interval
accuracy is dominated by negative bins and does not indicate useful contact detection.
The contact estimator needs improvement before freezing collision comparisons. Future
iterations must use new held-out groups rather than tune against this measured set.
The two variants within each scene are correlated; this small run has no confidence intervals.

## Remaining work

Improve contact timing on validation scenes and verify it on a fresh held-out set.
Then test extraction failures on actual model outputs. Action controls, wall-contact
scoring, image metrics, temporal consistency, ensemble diversity/coverage, and scene-group
aggregation still need implementation. T07 remains open.

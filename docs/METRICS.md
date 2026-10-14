# Evaluation implementation

T07 now has NumPy trajectory and object-object collision metrics, a JSON export command,
and tests with known outcomes. These consume extracted positions and contact onsets.
They do not yet track objects or infer contacts from generated pixels.

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

## Next integration work

Obtain a raw simulator fixture with frame-0 identities, masks, positions, timestamps,
and contact onsets from A. Validate a shared pixel tracker and contact estimator on held-out
simulator clips, including overlaps and failures, before scoring generated videos.

Action controls, wall-contact scoring, image metrics, temporal consistency, ensemble
diversity/coverage, and scene-group aggregation still need implementation. T07 remains open.

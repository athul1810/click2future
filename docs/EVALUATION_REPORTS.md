# Comparison tables

`click2future.evaluation.aggregate` combines the existing trajectory/contact, image,
and temporal/ensemble JSON reports. Run it separately for ID, each OOD suite, and
auxiliary-head outputs. Action-control reports still use their own protocol and
should be shown alongside these tables.

## Reproduce the control check

With the CPU and image-metrics extras installed:

```sh
python -m click2future.evaluation.aggregate_controls --output-dir outputs/table-controls --record outputs/table-controls-check.json
```

Use a fresh directory. The command generates four scenes, four hidden-physics
variants per scene, simulator-video and frozen-video controls, and 24 metric reports.
It writes `comparison.json`, `summary.json`, and `table.csv`. It checks that frozen
videos have zero trajectory spread and worse trajectory error than the simulator
futures. The small reproducibility record is in `results/aggregation-controls.json`.

These are correctness controls. The reference ensemble overlaps the simulator-video
inputs deliberately. Four scenes cannot establish model performance or reliable
population confidence intervals. No generator is trained by this command.

## Evaluate model runs

Prepare per-case reports with the existing `video`, `images`, and `futures` commands.
Use one run ID for one trained checkpoint and fixed sampling seeds. Declare the
expected cases before evaluating, then make a comparison manifest with these fields:

| Field | Contents |
|---|---|
| `schema_version` | `1` |
| `source` | `video_tracking`, `auxiliary_head`, `control_fixture`, or `simulator_control` |
| `dataset_id`, `protocol_id`, `suite` | Frozen dataset/protocol identifiers and the test suite |
| `sampling_seeds` | Ordered distinct seeds, one per sample; fixes K for every run |
| `cases` | Expected `scene_group_id`, `condition_id`, and single-reference `sample_id` |
| `cases[].reference_ids` | Ordered simulator reference set when using futures reports |
| `runs` | Method label, unique `run_id`, `training_seed`, `checkpoint_sha256`, and `reports` |
| `runs[].reports` | One entry per case, with `case_index` and report paths under `trajectory`, `images`, and/or `futures` |

Report paths are relative to the manifest. Use the same metric families for every
case/run. `comparison.json` from the control command gives a complete working example;
replace its control metadata with real checkpoint hashes and training/sampling seeds.
Controls do not require a trained checkpoint. Never relabel them as model results.

```sh
python -m click2future.evaluation.aggregate --manifest runs/comparison.json --output outputs/comparison.json --csv outputs/comparison.csv --bootstrap 2000 --seed 7
```

The exporter rejects missing/failed reports, duplicate cases, mismatched K, changed
settings, different frame timing, mixed metric sources, and different reference
inputs between runs. RGB metric families must use the same video and reference bytes.
Regenerate old trajectory reports that do not retain `timestamps_seconds`.
The manifest declares sample seed order; the exporter checks K and input hashes,
but cannot recover seeds from a video file. Retain generation manifests as evidence.

## Averaging and uncertainty

For scalar scores, average cases within a scene, then runs within that scene, then
independent scenes with equal weight. Multiple actions or physics variants of one
scene do not become independent observations. Repeated training seeds within a method
are rejected. Run/checkpoint IDs, per-case scores, and all metric counts remain in JSON.

The deterministic percentile bootstrap resamples whole scene groups, using the same
group draws for every method. Its default is 2,000 replicates and 95% intervals.
Intervals condition on the provided checkpoints and do not estimate training-seed
uncertainty. Separate training-seed variability must be reported when enough runs exist.

A null case/run score makes that scene unavailable for that metric. The output keeps
expected/scored scene counts and underlying missing-observation counts. With fewer
than two scored scenes, the scalar confidence interval is null. Compare missing rates
alongside conditional scores, especially when models fail on different scenes.

Collision scores pool TP/FP/FN/TN counts, then bootstrap those totals by scene.
This produces micro precision/recall/F1, interval accuracy, and positive prevalence.
Undefined ratios remain null; event-free cases never become perfect F1. The output
records how many bootstrap replicates had a defined ratio. The contact estimator still
has known extraction errors, so these tables do not make collision comparisons reliable
by themselves. Keep its simulator validation and missing-pair counts visible.

Mean ADE/FDE and whole-sample best-of-K remain separate columns. Table export does
not complete T14: the actual E0-E4 runs, budget measurements, hyperparameter study,
and analysis are still required.

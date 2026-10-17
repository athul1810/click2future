# Experiment protocol and batch evaluation

This implements the evaluation workflow for T13-T16. The core comparison is
E0/E2/E3 at 64px, 32 total frames and 8 FPS. Training results are still required.
The selected prediction target, final dataset/OOD versions, checkpoints and main
training budget remain open. The protocol cannot be called fully frozen yet.

`configs/evaluation/comparison.yaml` records common sampling seeds 0-3, independent
training seeds 0-2, metric configuration and scene-group bootstrap settings.
128px is deferred: the current diffusion pilots fail coherent 64px sampling, so
increasing resolution would spend compute before the core methods work.

Reserve a validation-only study of 16/32/64 DDIM steps on at most 16 conditions,
using the same checkpoint and four sampling seeds. Cap it at 0.5 GPU-hours.
Record quality, tracking failures, action adherence and latency. This is a budget
limit, not a measured cost or a claim about the account's total quota. Choose the
sampler on validation before final test evaluation. Retain every attempted setting.

## Prepare the cases before comparing checkpoints

```sh
python -m click2future.evaluation.suite prepare --dataset data/main --split validation --output outputs/validation-plan.json
python -m click2future.evaluation.suite prepare --dataset data/main --split test --output outputs/test-plan.json
```

Each plan includes every condition and all its hidden-physics variants in the
chosen split. It hashes the split, dataset index, reference files, extractor config
and comparison protocol, plus evaluator source hashes. Existing plans are never
overwritten. Changed inputs or metric code
require an explicit new plan and a recorded reason. Preparing test references does
not authorize test-based model selection.

For every case's `sample_id`, generate four futures using only its first frame,
selected-object mask and physical force, with seeds 0-3. Keep exact frame 0.
Use the same request for every method. Write uint8 `videos[K,T,H,W,3]` NPZ files.
An optional boolean `frame_valid[K,T]` marks missing RGB; missing observations
remain in trajectory, contact, image and uncertainty denominators.

## Evaluate the model runs together

Write a prediction manifest next to the outputs. Paths resolve relative to that
manifest. Replace the case keys with **all** `sample_id` values from the plan:

```json
{
  "source": "generated_video",
  "sampling_seeds": [0, 1, 2, 3],
  "runs": [
    {
      "method": "E2",
      "run_id": "E2-seed0",
      "training_seed": 0,
      "checkpoint": "runs/E2-seed0/checkpoint.pt",
      "videos": {
        "CASE_SAMPLE_ID": "samples/E2-seed0/CASE_SAMPLE_ID.npz"
      }
    }
  ]
}
```

Add each E0/E2/E3 run, with its own checkpoint and training seed. The evaluator
hashes checkpoint bytes and rejects incomplete case coverage, changed input hashes,
different sampling seeds, wrong K, changed frame 0 and incompatible frame timing.
It does not certify that a named checkpoint was trained successfully; retain its
training manifest and inspect the generated-video quality gate.

```sh
python -m click2future.evaluation.suite run --plan outputs/test-plan.json --predictions outputs/predictions.json --output-dir outputs/test-results
```

The new output directory contains per-case trajectory, image and futures reports,
`comparison.json`, and aggregate `summary.json`/`summary.csv`. All metric families
use the same video bytes. Whole-clip best-of-K and missing counts remain explicit;
uncertainty coverage uses all references. Failures produce `failure.json` without
a successful summary. Reports from a failed run must not be used in final tables.

Collision tables retain three separate views: any encounter per object pair over
the clip, exact interval labels, and one-to-one timed events. Report prevalence and
missing pairs alongside encounter accuracy. A high encounter score does not establish
correct timing or repeated contacts. The frozen extractor has separate stress checks
on controlled interaction scenes in `results/controlled-extractor-checks.json`.

Simulator controls use `source: simulator_control` and names such as `oracle` or
`static`, with no checkpoint fields. They cannot use E0-E4 method names. SSIM-only
development checks require an explicit `--metrics ssim` plan; final comparisons
request SSIM and LPIPS. Cached LPIPS weights stay outside the source ZIP.

## Remaining evidence

T14 needs actual trained E0/E2/E3 samples and repeats. T15 also needs Sanjay's
separate shape/texture/force OOD datasets. T16 needs generated samples compared
against repeated hidden-physics references and paired action controls. Run the
existing action evaluator on its controlled, contact-free windows separately.
This batch workflow does not replace those experiments or their quality checks.

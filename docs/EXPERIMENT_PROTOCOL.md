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

## Generate the samples from checkpoints

Edit `configs/evaluation/runs.example.json` to point to the actual frozen E0/E2/E3
checkpoints. Paths resolve relative to that JSON file. Add independent training
seeds as separate runs; the runner reads each seed and training step from the
checkpoint. Set DDIM steps to the validation-selected value before the test run.

```sh
python -m click2future.evaluation.sample_suite --plan outputs/test-plan.json --runs configs/evaluation/runs.example.json --output-dir outputs/model-samples --device cpu
python -m click2future.evaluation.suite run --plan outputs/test-plan.json --predictions outputs/model-samples/predictions.json --output-dir outputs/test-results
```

The runner covers every planned case with its four ordered seeds, using the existing
CVAE prior sampler or diffusion EMA sampler. It checks the training corpus/split
hashes, model names, normalization, force limits, dimensions and timing. Diffusion
runs must share target, schedule length and backbone width. Full training configs
and sampling times remain in `sampling.json`; these checks do not replace T13's
validation and budget decisions. Use explicit `--split id_test` when preparing a
dataset-0.3 ID plan, and its named OOD splits for separate plans.

CUDA sampling requires an allocated TC2 GPU job. `submission/sample_evaluation.sbatch`
runs sampling and all three metric families with `C2F_EVAL_PLAN`, `C2F_EVAL_RUNS`,
`C2F_SAMPLE_DIR`, `C2F_RESULT_DIR` and `C2F_SOURCE_REVISION` set. The plan and run
specification are local files; the script does not train or select models.

Existing output folders are rejected. Partial sampling writes a failure record
without a complete prediction manifest. Checkpoint/video hashes are bound to that
manifest, so overwriting an artifact after generation makes evaluation fail.
Untrained checkpoints are refused unless `--allow-untrained-fixtures` is explicitly
set for a correctness check; zero-step runs remain labelled in the summary JSON.
E4 uses generated RGB here; auxiliary heads retain their separate evaluation API.

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

## Paired zero-force and reversed-force checks

For dataset-0.2, freeze the action configuration when preparing the plan:

```sh
python -m click2future.evaluation.suite prepare --dataset data/controlled --split test --actions-config configs/evaluation/actions.yaml --output outputs/action-test-plan.json
```

Use that plan in the same sampling and evaluation commands above. In addition to
the trajectory/image/futures tables, evaluation writes `actions.json` and
`actions.csv` from the same four ordered seeds. It reports direction, zero-force
drift, reversal/sign scores, stationary outputs and missing tracks. Optional
`frame_valid` flags apply to the fixed action window too.

Eligibility is frozen from references before evaluating models. The first 0.5
seconds must be contact-free in every forward/reverse hidden-physics variant.
Groups with reference contacts remain listed with an exclusion reason. Missing
actions, changed targets, mismatched physics or timing reject plan preparation.
The zero reference starts at rest and shares physics with one reversal pair;
all other forward/reverse physics variants must match exactly.

Summaries average within each scene and training run, then across scene groups.
Confidence intervals resample whole scene groups; one group has no interval.
Conditional rates retain their complete/missing counts in JSON and missing rates
in the CSV. An entirely excluded suite has no scored action metrics.

The same outputs include separate simulator-label and rendered-reference scores
over all matched physics variants. Some low-force references stay below the fixed
motion threshold, so a low sign-response rate alone does not establish a physics
failure. Compare it with these reference scores, tracking rates and trajectory
errors. Reference trajectories are calibration data only; they never condition
model sampling.

Dataset-0.3 validation, ID, shape, texture and repeated-physics splits support the
same checks. Its force-OOD split deliberately omits zero force, so evaluate that
split with the regular metrics and no `--actions-config`. Existing target-change
conditions also reverse the force, so they do not isolate selected-object changes.

## Selected-object controls with the same force

Derive a separate intervention plan from an ordinary frozen held-out plan:

```sh
python -m click2future.evaluation.targets --plan outputs/test-plan.json --output-dir outputs/target-controls
python -m click2future.evaluation.sample_suite --plan outputs/target-controls/plan.json --runs configs/evaluation/runs.example.json --output-dir outputs/target-samples --device cpu
python -m click2future.evaluation.suite run --plan outputs/target-controls/plan.json --predictions outputs/target-samples/predictions.json --output-dir outputs/target-results
```

The recipe uses every scene group's forward condition. For each hidden-physics
reference it replays the same image, force, duration and physical objects, selecting
the original target and then the lowest-index other valid object. Only the clicked
object changes. It keeps all reference variants and writes them outside the training
corpus. The original manifest and split hashes remain the checkpoint compatibility
checks. The intervention plan binds the parent plan, original inputs, simulator
revision, generated references and fixed action settings. Changing an input rejects
sampling or evaluation; use a fresh directory for every preparation.

This suite is labelled `SPLIT-selected-object` and stays separate from ordinary
ID/OOD tables. It does not establish generalization to a different visual corpus.
Sources without a forward condition, such as dataset-0.1, are rejected.

The optional `--force-magnitude 0.4` sets one shared magnitude in the source forward
direction for both clicks and records it in the plan. Without it the source force
is preserved. Choose any override using validation before trained-model test
evaluation; never choose a magnitude per scene or retune the motion threshold.
The first validation correctness check at force 0.15 had all eight exact-label
pairs below threshold, while rasterized references scored differently. That record
is retained alongside a separate 0.4 control check in
`results/selected-object-control-checks.json`. Neither is a learned-model result.

The same checkpoint runner generates four paired futures for each condition.
Evaluation writes the ordinary trajectory/image/futures reports plus `targets.json`
and `targets.csv`. A successful target pair requires the selected object to move
at least one pixel along the force and every unselected object to remain below
the one-pixel drift threshold throughout the first 0.5 seconds, in both conditions.
A separate contrast checks that each object's force projection increases by at
least one pixel when it is clicked. Identical motion under both clicks fails that
contrast. Stationary, missing and unselected-object drift counts remain explicit.
Both conditions require complete tracks of every valid object in this window.

Eligibility uses only simulator references: every physics variant under both
clicks must start at rest and avoid contact within the fixed window. Excluded groups
stay in the plan with reasons. No group is excluded for weak reference motion or
poor model output. Exact-label and rendered-RGB reference calibration, paired
sample seeds and scene-group confidence intervals follow the action protocol above.
The first-frame-only E1 control must produce identical videos under both clicks
with paired seeds; it cannot observe the selection.

## Remaining evidence

The [multi-scene pilot](MULTISCENE_PILOT.md) now measures actual E0/E2/E3/E4
validation ensembles, repeated-physics coverage and both action-control families.
Its [result record](results/multiscene-validation-checks.json) retains failures,
image intervals, E4 heads and the 16/32/64-step study. The original models fail
complete trajectory extraction; the flow decoder improves appearance but remains
stationary under the action controls without displacement labels. With those
labels, only 1/16 reversal pairs and 0/16 selected-object pairs pass.

T14 still needs useful-checkpoint comparisons and independent repeats. T15 keeps
all 248 ID/OOD/physics test clips reserved while validation fails. T16 still needs
an ensemble that supports physical uncertainty measurements. Completing a batch
workflow or observing RGB variation does not satisfy these quality checks.

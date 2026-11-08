# Run: D019-D021 physical quality investigation

| Field | Value |
|---|---|
| Status | Complete; both neural repair candidates rejected |
| Owner | Unassigned |
| Training source revisions | D019: 2904d57232afa458af2bfa9effa031d7bc511ebf; evaluation recovery: 612dedc2e6aa9ea5a21fdc5ce0549ed5acb9390e; D020: f4fea11fd6a278b16e3ba917d28090eae594c827 |
| D021 generator/scorer freeze | cb3aa2e3721d6c7c93cedec5707b1e6d8d488d67, before corpus generation |
| Environment | Python 3.12.15; training PyTorch 2.11.0+cu128; pinned project lock |
| Training hardware | TC2, one NVIDIA A40 per job; 4 CPU, 24 GB RAM |
| Configs | configs/training/prior_expected_control.json, prior_expected_expected.json, prior_rest_anchored.json |
| Training seed / final updates / batch | 19 / 8,000 / 64 for every arm |
| Initialization | Same E7 EMA teacher; fresh optimizer and private RNG |
| Teacher SHA-256 | 93db6c714a0049b311117abe68faeef0d0e27080315bb8e67a2ecda59010c041 |
| Cache manifest SHA-256 | 2c695895d5caed7be0369b28827739aab677a127c00b3d85b6e9f2488f9b9290 |
| Selection | Fixed final update, no best-checkpoint selection |
| Model size | D020 1,064,580 parameters; physics baseline has no learned parameters |
| D020 checkpoint SHA-256 | 60e01fd60e6934e644653662d701a8aa8419d57d3d4d58a65d8b6ca87a3a45e4 |
| Input/output | 64px; 32 frames at 8 fps; four worlds per condition |
| D021 data seed / groups / conditions | 910000 / 10 fresh ID groups / 60 conditions |
| Sampling seeds | 0, 1, 2, 3, retained for every condition |
| Allocated training/evaluation time | Jobs 50745, 50747 and 50749: 24 minutes 2 seconds total |
| Training peak CUDA allocation | D020: 636,747,264 bytes |

## Reproduction and recovery

Use the fixed configs with `submission/expected_prior_comparison.sbatch` and
`cluster/submit_once.py`. D019 job 50745 completed training but failed its first
format-8 audit. Preserve the original attempt. D019 evaluation-only job 50747
uses `C2F_TRAINED_RUN` to reuse those retained checkpoints, without retraining.
D020 job 50749 uses `C2F_REST_ANCHORED=1`. Each arm saves optimizer and RNG state
every 250 updates. CUDA continuation probes had zero model/EMA tensor error.
Detached, idempotent Slurm submission and retained remote artifacts tolerate VPN
loss. A disconnected client cannot make a submitted job disappear; a real job
failure remains a failure and is reported.

For D021, use `python -m click2future.evaluation.physics_screen --fresh-id
--dataset FRESH_DATA --training-data ORIGINAL_DATA --checkpoint D020_FINAL
--output FRESH_SCREEN`. Keep the frozen plan and all raw outputs. The physics
generator reads only the image, selected mask, force and sampled prior worlds.
Reference scene recipes enter evaluation only.

## Results and limits

See [the measured results and interpretation](../docs/PHYSICAL_QUALITY_REPAIR.md)
and [the compact machine-readable record](../docs/results/physical-quality-d021.json).
D019 did not fix rest and worsened penetration. D020 fixed rest by architecture
but still penetrated in validation and fresh scenes. The optional physics
baseline passed measured penetration/rest checks, while both generators failed
the original strict single-reference trajectory gate. Contact timing and
uncertainty remain unresolved. Ten ID groups do not establish OOD reliability.

The matched-world analysis was specified after the primary scores and is
exploratory. It retains all predictions and does not replace primary metrics.
The rendered-reference contact control exposes error in the existing RGB event
estimator. No metric, threshold, checkpoint or generator was tuned on D021.

Browser evidence checks all four generated futures, exact rest, image-only
inputs, explicit method labels, comparison with two neural models, mobile layout
and JavaScript errors. Nine checks passed. Raw outputs, screenshots, logs,
frozen records and remote backups are kept locally outside Git. They are not
included in submitted artifacts.

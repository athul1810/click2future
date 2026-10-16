# Verification

The current tests cover the simulator/data scaffold:

- deterministic scene generation and hidden-physics variants;
- force direction/reversal, zero-force behavior, and known contact events;
- deterministic RGB and instance-mask rendering;
- rollout shapes, timestamps, action maps, and oracle labels;
- JSON/NPZ round trips and schema validation.
- saved-rollout replay and mismatch detection.
- model-facing channel-first sample and padded batch loaders.
- the fixed scene fixture and end-to-end generate/validate CLI flow.

`test_environment.py` checks the CPU diagnostic/report and the failure status for an
unavailable CUDA device. Run `python -m pytest -q` using the environment in
[development setup](../docs/SETUP.md). `test_pilot_denoiser.py` checks forward/backward,
auxiliary shapes/symmetry, E1 input isolation, equal-capacity E2/E3 and zero-force
agreement, spatial coordinate recovery, invalid inputs, and exact CPU optimizer/RNG restart continuity.
`test_profile_pilot.py` verifies the allocation guard, YAML settings and CLI overrides,
resolved report settings, precision validation, and failure on non-finite gradients. These are prototype checks;
learned action response, sampling, and full trainer tests remain pending.

`test_evaluation_metrics.py` validates analytical trajectory and collision controls.
`test_evaluation_cli.py` checks source-labelled JSON reports, metadata alignment, corrupt
inputs, and input-file preservation. `test_pixel_tracking.py` covers identity crossings, missing/duplicate objects, colour
ambiguity, contact onsets, and separation of pixel extraction from future oracle labels.
`test_extractor_calibration.py` checks validation selection and failure denominators.

Run `python -m click2future.evaluation.calibrate` with the output arguments in
[METRICS.md](../docs/METRICS.md) to reproduce the simulator extraction measurements.
These measure the extractor, not a trained model.

`test_action_controls.py` checks direction/reversal, zero-force drift, stationary and
missing outputs, paired reference validation, and both simulator/generated-video CLI paths.
Contact tests cover opposing-motion peaks, near-pass rejection, and missing context.

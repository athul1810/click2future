# Experiments

Store a short Markdown manifest per meaningful run using [RUN_TEMPLATE.md](RUN_TEMPLATE.md). Suggested name: `YYYYMMDD_E2_seed0.md`.

The experiment IDs and comparison rules are in [the evaluation plan](../docs/EVALUATION.md).
TC2 diffusion diagnostics are recorded in `docs/results/diffusion_pilot_2026-10-16.json`
and `diffusion_debug_2026-10-16.json`; both failed generated-video quality checks.
Main E0/E2/E3 comparisons remain open. Use the [batch evaluation workflow](../docs/EXPERIMENT_PROTOCOL.md)
for fixed cases and matched reports. Keep large logs and checkpoints outside Git.

These manifests are team working records. Do not include external artifact links or checkpoints in the submitted source-code ZIP; use local configs, seeds, and source instructions instead.

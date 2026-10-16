# Configurations

- `simulator/smoke.toml`: short simulator checks.
- `simulator/pilot.toml`: frozen 32-frame, 64px simulator setting.
- `development.yaml`: artifact-directory conventions.
- `pilot.yaml`: synthetic denoiser profiler dimensions, conditioning, heads, object count,
  and precision. This is a prototype configuration, not a dataset trainer.
- `evaluation/pixel.yaml`: 64px pixel tracker and validation-selected contact gap.
  Contact event accuracy is currently too low for model comparisons; see
  [the measurements](../docs/METRICS.md).

The simulator reads TOML; the profiler and pixel evaluator read YAML. Save resolved
settings with each run. T13 will freeze the model comparisons after validation.

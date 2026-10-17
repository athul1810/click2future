# Configurations

- `simulator/smoke.toml`: short simulator checks.
- `simulator/pilot.toml`: frozen 32-frame, 64px simulator setting.
- `simulator/dataset_0_2.toml`: controlled zero, directional, object-contact, and wall dataset.
- `training/pilot_diffusion_0_3.json`: separate generalization-corpus diagnostic with
  a declared 1.5 force limit; see [the data guide](../docs/GENERALIZATION_DATA.md).
- `development.yaml`: artifact-directory conventions.
- `pilot.yaml`: synthetic denoiser profiler dimensions, conditioning, heads, object count,
  and precision. This is a prototype configuration, not a dataset trainer.
- `evaluation/pixel.yaml`: 64px/8-FPS pixel tracker and validation-selected motion/contact settings.
  Contact event accuracy is currently too low for model comparisons; see
  [the measurements](../docs/METRICS.md).

The simulator reads TOML; the profiler and pixel evaluator read YAML. Save resolved
settings with each run. T13 will freeze the model comparisons after validation.

`evaluation/actions.yaml` fixes the early contact-free window and motion threshold for
paired action controls. Use the commands in [METRICS.md](../docs/METRICS.md).

`training/auxiliary.yaml` configures the loss adapter and low-timestep gate. Its values
are untuned proposals; see [the integration guide](../docs/AUXILIARY_LOSSES.md).

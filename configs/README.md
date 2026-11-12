# Configurations

- `simulator/smoke.toml`: short simulator checks.
- `simulator/pilot.toml`: frozen 32-frame, 64px simulator setting.
- `simulator/dataset_0_2.toml`: controlled zero, directional, object-contact, and wall dataset.
- `training/pilot_diffusion_0_3.json`: separate generalization-corpus diagnostic with
  a declared 1.5 force limit; see [the data guide](../docs/GENERALIZATION_DATA.md).
- `training/prior_direct_contact_dynamics.json`: direct-prior candidate with optional
  contact-transition and normal-velocity losses; `prior_direct.json` is its zero-weight control.
- `training/prior_rest_anchored_geometry_x{16,128,1024}.json` and
  `training/prior_direct_contact_dynamics_x{128,1024}.json`: D032 copies of the D020 and
  D029 recipes with only the geometry or contact-dynamics weights scaled; see
  [the study](../docs/GEOMETRY_LOSS_SCALE.md).
- `training/prior_direct_seed{20,21}.json` and
  `training/prior_direct_contact_dynamics_x1024_seed{20,21}.json`: D034 copies that change
  only the training seed.
- `training/prior_rest_anchored_dynamics_residual.json`: D037 candidate that adds a
  free-motion force/mass acceleration residual to the rest-anchored expected prior.
- `training/prior_rest_anchored_seed20.json` and
  `training/prior_rest_anchored_dynamics_residual_seed20.json`: the matched seed-20
  control and candidate registered for the D038 full-pipeline comparison.
- `training/collision_impulse_screen_control.json` and
  `training/collision_impulse_screen_x1024.json`: matched 2,000-update seed-22 recipes
  for the exploratory D039 collision-impulse screen.
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

`evaluation/runs.example.json` declares local checkpoint paths and DDIM steps for
the fixed-case sampling runner. Edit paths and validation-selected steps before use;
see [the experiment protocol](../docs/EXPERIMENT_PROTOCOL.md).

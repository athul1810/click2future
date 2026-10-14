# Planned implementation

The package provides `python -m click2future.env_check`, an untrained
`models.pilot_denoiser.PilotDenoiser`, and `python -m click2future.profile_pilot` for
synthetic architecture profiling. `python -m click2future.evaluation` scores extracted
trajectories and contact events; see [metric usage](../../docs/METRICS.md). It has no
simulator, dataset training, or sampling
implementation yet. See [the build plan](../../docs/BUILD_PLAN.md). Planned modules:

| Module | Responsibility | Tasks |
|---|---|---|
| `sim/` | Physics, rendering, deterministic replay | T05 |
| `data/` | Export, split manifests, datasets and batching | T04, T06 |
| `models/` | Autoencoder, CVAE, video denoiser, conditioning and heads | T08-T12 |
| `training/` | Losses, optimization, checkpoints and sampling | T08-T13 |
| `evaluation/` | Tracking, metrics, controls and result export | T07, T14-T16 |

Implement against [the data contract](../../docs/ARCHITECTURE.md). Add tested setup and CLI instructions through T03 as functionality becomes available.

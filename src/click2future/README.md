# Planned implementation

The package currently provides `python -m click2future.env_check` for development
environment diagnostics. It has no simulator or training implementation yet. Proposed modules:

| Module | Responsibility | Tasks |
|---|---|---|
| `sim/` | Physics, rendering, deterministic replay | T05 |
| `data/` | Export, split manifests, datasets and batching | T04, T06 |
| `models/` | Autoencoder, CVAE, video denoiser, conditioning and heads | T08-T12 |
| `training/` | Losses, optimization, checkpoints and sampling | T08-T13 |
| `evaluation/` | Tracking, metrics, controls and result export | T07, T14-T16 |

Implement against [the data contract](../../docs/ARCHITECTURE.md). Add tested setup and CLI instructions through T03 as functionality becomes available.

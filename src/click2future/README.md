# Click2Future implementation

The package includes simulation, grouped data generation/replay, an untrained denoiser
prototype, environment/profiling commands, and evaluation. Follow
[development setup](../../docs/SETUP.md) for the full environment. `requirements.txt`
is a smaller simulator-only environment. Dataset training and sampling remain to be built.

Current modules:

| Module | Responsibility | Tasks |
|---|---|---|
| `sim/scene.py` | Deterministic scene sampling | T05 |
| `sim/backend.py` | Pymunk physics adapter and contacts | T05 |
| `sim/render.py` | Headless RGB and instance-mask rendering | T05 |
| `sim/rollout.py` | Fixed-step clip assembly and oracle labels | T05 |
| `data/export.py` | JSON plus compressed NumPy I/O | T06 |
| `data/loader.py` | Channel-first sample/batch tensors | T06 |
| `data/validate.py` | Rollout schema checks | T04, T06 |
| `data/splits.py` | Scene-group manifests and hashes | T06 |
| `cli.py` | Smoke, grouped dataset, and split commands | T03, T06 |
| `sim/replay.py` | Saved-rollout regeneration and comparison | T05, T06 |
| `models/` | Untrained pilot denoiser, conditioning and heads; autoencoder/CVAE pending | T08-T12 |
| `training/` (planned) | Dataset training, checkpoints and sampling | T08-T13 |
| `evaluation/` | Tracking, metrics, controls and result export | T07, T14-T16 |

Implement against [the data contract](../../docs/ARCHITECTURE.md).
`python -m click2future.evaluation.video` extracts and scores RGB tracks and contact
estimates. See [usage and measured limitations](../../docs/METRICS.md).

The grouped generator is available from the repository root with:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m click2future.cli generate-dataset `
  --output-dir outputs/pilot_32f `
  --group-count 250 `
  --variants-per-group 4 `
  --seed 0 `
  --config configs/simulator/pilot.toml
```

This command creates one split assignment per visible scene group and writes
`dataset_manifest.json`, `splits.json`, and the rollout archives. It is not
run as part of setup; choose the counts explicitly before starting generation.

Validate a saved sample, including deterministic replay, with:

```powershell
python -m click2future.cli validate-sample `
  --input-dir outputs/pilot_32f `
  --sample-id scene-00000032-variant-001 `
  --replay
```

Downstream code can use `click2future.data.load_sample` for one normalized
`[T,3,H,W]` sample or `click2future.data.load_batch` for padded
`[B,T,3,H,W]` batches.

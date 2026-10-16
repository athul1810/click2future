# Workstream A: Simulator and Data

Workstream A provides the deterministic simulator, rendered clips, physical
labels, dataset splits, loaders, and replay validation used by downstream
models and evaluation.

## Implemented scope

- 2D top-down world with fixed boundaries and zero gravity
- 2-4 circles or rectangles per scene
- randomized position, geometry, color, mass, friction, and restitution
- one centre-applied force on a selected object
- deterministic Pymunk simulation and Pillow rendering
- RGB frames, instance masks, positions, and contact-onset labels
- JSON metadata plus compressed NumPy arrays
- grouped hidden-physics variants and leakage-safe splits
- exact replay from saved scene metadata
- channel-first sample and padded batch loaders

The simulator runs on CPU. GPU resources are reserved for downstream model
training.

## Stack

- Python 3.12
- NumPy 2.5.4
- Pillow 12.3.0
- Pymunk 7.3.1
- TOML configuration through Python's `tomllib`
- `argparse` command-line interface

Dependencies are pinned in `requirements.txt`.

## Pilot configuration

`configs/simulator/pilot.toml` defines the current pilot:

- world size: 1 × 1
- resolution: 64 × 64
- frames: 32
- output rate: 8 FPS
- physics rate: 240 Hz
- force duration: 0.25 seconds
- objects: 2

Each visible scene has four variants with the same geometry, appearance, frame
0, selected object, and force. Only hidden physics changes between variants:

- mass: uniform from 0.5 to 2.0
- friction: uniform from 0.1 to 0.8
- restitution: uniform from 0.0 to 0.4

All randomness comes from recorded seeds. A scene group and all of its physics
variants always belong to the same split.

## Sample contract

Each saved rollout contains:

| Field | Shape or type |
|---|---|
| `frames` | `uint8 [T,H,W,3]` |
| `instance_masks` | `bool [T,N,H,W]` |
| `object_valid` | `bool [N]` |
| `selected_object_id` | integer |
| `selected_mask` | `bool [H,W]` |
| `force_xy` | `float32 [2]` |
| `action_map` | `float32 [2,H,W]` |
| `timestamps` | `float64 [T]` |
| `positions` | `float32 [T,N,2]` |
| `collision_onset` | `bool [T-1,N,N]` |
| `wall_contact_onset` | `bool [T-1,N,4]` |
| `scene_spec` | replayable scene and hidden physics metadata |
| `metadata` | simulator, renderer, and resolved timing settings |

Raw samples use `[T,H,W,3]`. `load_sample` converts frames to normalized
`[T,3,H,W]`; `load_batch` returns normalized `[B,T,3,H,W]` and pads object
slots with `object_valid`.

## Commands

Generate the grouped 32-frame pilot:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m click2future.cli generate-dataset `
  --output-dir outputs/pilot_32f `
  --group-count 250 `
  --variants-per-group 4 `
  --seed 0 `
  --config configs/simulator/pilot.toml
```

Validate and replay one sample:

```powershell
python -m click2future.cli validate-sample `
  --input-dir outputs/pilot_32f `
  --sample-id scene-00000032-variant-001 `
  --replay
```

Run the tests:

```powershell
python -m pytest -q
```

Generated datasets remain under `outputs/` and are excluded from Git.

## Boundaries

This dataset measures planar motion, action adherence, collisions, wall
contacts, and variation caused by hidden physical properties. It does not model
3D depth, stacking, lifting, photorealism, or transfer to real video.

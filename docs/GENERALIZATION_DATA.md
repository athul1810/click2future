# Generalization data

Dataset-0.3 is a separate matched corpus for the OOD experiments in T06/T15.
It extends the controlled two-object layouts with object-attached textures and
physical triangles. Dataset-0.1 and dataset-0.2 generation remain unchanged.
This is an available data path, not the frozen training dataset or a model result.

Train and select models on this corpus before comparing its ID and OOD suites.
Mixing its test results with models trained on dataset-0.2 would also change the
training texture distribution. Keep those experiments separate.

| Split | Shapes | Textures | Nonzero force magnitudes | Physics variants |
|---|---|---|---|---:|
| `train` | circles, rectangles | solid, stripes | 0.15, 0.25, 1.1 | 4 |
| `validation` | same as train | same as train | same as train | 4 |
| `id_test` | same as train | same as train | same as train | 4 |
| `shape_ood` | triangles | same as train | same as train | 4 |
| `texture_ood` | same as train | checker | same as train | 4 |
| `force_ood` | same as train | same as train | 1.3 or 1.5 | 4 |
| `physics_test` | same as train | same as train | same as train | 8 |

All seven splits have disjoint visible seeds and scene groups. Every action and
hidden-physics variant from one group stays in that split. Only the declared
factor changes in each OOD definition. Background, colours, layout distribution,
mass/friction/restitution ranges, timing and simulator settings stay shared.
Physics-test conditions have more draws from the same hidden-property ranges;
they are an uncertainty test, not physics-OOD.

Each ordinary group contains the six existing actions and 21 clips. Force-OOD
excludes zero force, giving 20 clips; zero would be an ID action. Each repeated-
physics group has 41 clips. The canonical zero clip is not duplicated. Compare
ID/OOD results by action family as well as in aggregate because force-OOD lacks
zero and the uncertainty split has more reference futures.

Patterns use two colours separated by +12 RGB levels, with a 0.04-world-unit
period. Stripes and checker cells move and rotate with the object. They do not
alter collision geometry or masks. This tests modest synthetic texture shifts,
not natural-image appearance transfer. Triangles use the same polygon vertices
for rendering and physics and fit within the original objects' bounds. Shapes
have different area and inertia; these are part of the shape shift.

## Generate and validate

```sh
python -m click2future.cli generate-generalization-dataset --output-dir outputs/generalization-smoke --seed 200000 --groups-per-split 1
python -m click2future.cli validate-generalization-dataset --input-dir outputs/generalization-smoke --replay
```

The smoke has seven groups and 166 native 64px, 32-frame clips. The Python API
`generate_generalization_dataset(..., group_counts={...})` permits different
positive counts for all seven splits. Record the full counts, seed and manifest
hash before a run; changing counts also changes subsequent split seeds.
Existing nonempty output folders are rejected. Generated data stays outside Git
and the final source ZIP.

The dataset manifest contains every sample's bytes hashes and the split-file
hash. Validation checks complete action/physics coverage, regenerated scene
specifications, identical initial frames within groups, unique split membership
and unchanged saved bytes. Optional replay compares all arrays and metadata
exactly. Legacy replay remains a separate required regression check.

`configs/training/pilot_diffusion_0_3.json` selects 21 training clips and accepts
physical forces up to 1.5, including the predeclared force-OOD inputs. This bound
does not add OOD clips to training. All forces still use the shared divisor 0.5,
without clipping. Use a separate run with recorded settings for dataset-0.3;
changing dataset versions is not an exact resume. Train indices use only `train`;
the CVAE also uses `validation` for its existing selection rule.

The batch evaluator can freeze each held-out split separately with `--split
id_test`, `shape_ood`, `texture_ood`, `force_ood` or `physics_test`. Keep the same
checkpoint, extractor and sampling settings across those plans. The current
pixel tracker is approximate; include failures when shapes, textures, collisions
or similar colours make it abstain. No trained OOD result is claimed here.

## Checks

An independent 14-group, 332-clip native smoke passed exact replay. The same code
also replayed 80 dataset-0.1 and 84 dataset-0.2 saved clips exactly, including
metadata. See [data checks](results/generalization-data-checks.json).

The frozen extractor from PR #30 was run without threshold changes on these
simulator clips. It tracked all future observations in validation, ID, shape,
force and repeated-physics controls, but only 1,953/2,604 in the texture control.
This 25% missing rate remains a limitation. The two groups per split are too
small to establish a general failure rate, and contact timing remains approximate.
[The record](results/generalization-extractor-controls.json) retains all failures
and separate event, interval and occurrence counts. These are simulator controls.

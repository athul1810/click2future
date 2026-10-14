# Workstream A: Simulator and Data Plan

Workstream A owns the physical world that every model and evaluation result depends on:

1. the deterministic tabletop simulator,
2. the rendered clips and oracle labels,
3. versioned, leakage-safe dataset splits, and
4. replay and validation tooling.

This plan turns the proposed interfaces in [ARCHITECTURE.md](ARCHITECTURE.md) into an implementation sequence. It intentionally does not cover model training or the interactive demo except where they consume Workstream A interfaces.

## Outcome

At the end of Workstream A, a teammate should be able to run one documented command that:

- creates a seeded tabletop scene,
- applies a documented action to a selected object,
- renders a reproducible video clip,
- exports masks, trajectories, and collision labels,
- saves enough metadata to replay the exact rollout,
- assigns the rollout to a leakage-safe split, and
- loads the result through one stable dataset interface.

The implementation must work on CPU for development and smoke tests. GPU access is not required for the simulator or dataset generator.

## Hardware-adjusted implementation choice

The available development machine has an RTX 4060 with 8 GB VRAM and 32 GB RAM. Workstream A will therefore be CPU-first and streaming by design. The GPU is reserved for the later compact model experiments; simulator and clip generation should not consume VRAM.

### Selected stack

- **Physics:** pinned Pymunk 7.x, backed by Chipmunk2D. It provides `Space.step()` for fixed-step simulation, rigid-body circle/segment/polygon shapes, friction/elasticity, and collision callbacks. We will wrap it behind our own `PhysicsBackend` protocol so the rest of the project does not depend directly on Pymunk APIs.
- **Rendering:** Pillow `Image`/`ImageDraw`, with no display server or game loop. RGB frames and per-object masks are rendered from the same object poses and geometry. RGB may use deterministic supersampling; masks remain binary and are validated separately.
- **Arrays and randomness:** NumPy arrays and `numpy.random.Generator` with `PCG64`/`SeedSequence`. Every scene gets explicit child random streams for geometry, appearance, hidden physics, action, and rendering.
- **Configuration:** TOML loaded with Python's standard-library `tomllib`; resolved configurations are serialized into run metadata and hashed.
- **Initial storage:** JSON metadata plus compressed NumPy `.npz` samples for fixtures and smoke data. After profiling, shard the same logical schema into a streaming format only if file count or throughput requires it.
- **CLI:** Python standard-library `argparse` initially, keeping the first scaffold small and dependency-light.

This stack is deliberately headless. Pygame, a GUI renderer, and a differentiable physics engine are not required for the first dataset. The simulator is an offline data generator, not an interactive game.

## Locked first-release scope: 2D top-down benchmark

The first release is intentionally a **2D top-down tabletop benchmark**, not a general 3D physics simulator. This is sufficient for the course comparison because it lets us measure action adherence, collision behavior, hidden-physics uncertainty, and generative diversity while keeping labels exact and the compute budget realistic.

Each object has planar state:

```text
(x, y, angle, vx, vy, angular_velocity)
```

The world contains bounded walls, zero or documented gravity, rigid objects, friction, mass, restitution, and centre-applied forces. Supported geometry starts with circles, rectangles, and convex polygons. The renderer is top-down and orthographic.

This benchmark can establish:

- whether generated motion follows the specified force;
- whether generated clips preserve object identity and geometry;
- whether collisions and wall contacts are plausible;
- whether samples vary with hidden mass/friction while sharing the same visible condition;
- whether spatial action conditioning and auxiliary supervision improve those behaviors.

It cannot establish photorealistic video quality, 3D contact dynamics, stacking/lifting, depth occlusion, or transfer to real photographs. Those limitations must be stated in the report. A 2.5D/3D extension is a later stretch goal and must not delay the 64x64 core comparison.

## Component boundaries

Keep the implementation split into five narrow components:

1. **Scene sampler** — produces a serializable `SceneSpec` and independent random streams. It knows distributions and rejection rules, but not model code.
2. **Pymunk backend** — converts `SceneSpec` into a physics space, advances fixed substeps, applies the centre force, and exposes poses/contact callbacks through our own protocol.
3. **Raster renderer** — converts a backend snapshot and render assets into RGB plus one binary mask per object. It has no physics side effects.
4. **Rollout assembler** — coordinates timestamps, frame-0 capture, action timing, substep event aggregation, normalized positions, and `Rollout` creation.
5. **Dataset layer** — validates the rollout, writes JSON/NPZ, creates scene-group manifests, and provides the downstream loader.

The model and evaluation code should consume `Rollout`/dataset records, not Pymunk objects. This prevents a physics-library change from changing the model interface.

### Minimal backend protocol

The first implementation should make the following operations explicit:

```text
build_space(scene_spec, simulator_config) -> BackendState
apply_force(state, object_id, force_xy, duration) -> None
step(state, dt) -> None
snapshot(state) -> BackendSnapshot
contact_events(state) -> ContactEvents
```

`PymunkBackend` is the only implementation initially. The protocol exists for testing and future replacement, not because we need multiple engines now.

### Minimal renderer protocol

```text
render_rgb(snapshot, render_spec) -> uint8[H,W,3]
render_instance_masks(snapshot, render_spec) -> bool[N,H,W]
```

Use the same object geometry and pose for RGB and masks. Do not infer masks from the rendered RGB frames; oracle masks should be exact simulator outputs.

### Why this fits the machine

Physics and rendering are low-resolution CPU work and can run one scene at a time, so 32 GB RAM is enough if clips are streamed rather than accumulated in memory. The 8 GB GPU remains available for the compact 64×64 latent video models. We will not commit to a dataset size or model budget until the smoke generator and a 64×64 training profile have been measured.

## Scope boundaries

### In scope

- 2D top-down bounded tabletop physics
- 2–4 rigid objects
- deterministic seeded scene generation
- fixed-centre force actions
- deterministic rendering at 64×64 initially
- per-object masks and object-slot identities
- object trajectories and wall/object collision events
- scene replay and version metadata
- dataset export, manifests, and split validation
- fixtures and tests for all of the above

### Explicitly out of scope for the first pass

- video models, autoencoders, or training loops
- photorealistic rendering or real photographs
- off-centre force/torque
- general-purpose 3D physics
- friction-conditioned model inputs
- the web/demo application
- packaged datasets or checkpoints in Git

## Contract gates before implementation

Before writing the physics backend, resolve and record these choices in [ARCHITECTURE.md](ARCHITECTURE.md) and [DECISIONS.md](DECISIONS.md):

- physics backend and version
- world units and pixel transform
- timestep, substeps, and integrator settings
- force units, magnitude range, and force duration
- friction and restitution ranges/law
- object shape families, size ranges, and appearance families
- frame count, FPS, and whether frame 0 is pre-action
- object-slot ordering and padding rules
- collision-onset and wall-contact definitions
- archive format and schema versioning strategy

For the first implementation, use Pymunk collision callbacks to observe contact begin/separate events, but also compute a post-step contact snapshot for validation. The exported label is an onset at a recorded-frame interval; a persistent contact must not become repeated onsets.

The contract should be finalized before generating the main dataset. Proposed values may be used for fixtures, but fixtures must be versioned when the contract changes.

## Proposed repository scaffold

The first implementation PR should create the package seams and a tiny fixture without pretending that the simulator is complete:

```text
src/click2future/
  __init__.py
  sim/
    __init__.py
    config.py          # simulator and renderer configuration types
    scene.py           # scene/object/action specifications
    backend.py         # physics-backend protocol/adapter
    rollout.py         # rollout state and exported labels
    render.py          # deterministic frame and mask rendering
    events.py          # contact onset and wall-contact extraction
    generate.py        # seeded scene/rollout generation
    replay.py          # replay from saved metadata
  data/
    __init__.py
    schema.py          # sample schema and schema version
    export.py          # archive writer
    load.py            # dataset reader and batching adapter
    splits.py          # scene-group split generation and checks
    validate.py        # schema/leakage/integrity validation

configs/
  simulator/
    smoke.toml
    default_64px.toml
  dataset/
    smoke.toml

tests/
  fixtures/
    tiny_scene.json
    expected_schema.json
  sim/
    test_determinism.py
    test_action_semantics.py
    test_collisions.py
    test_replay.py
  data/
    test_schema.py
    test_splits.py
    test_leakage.py
    test_roundtrip.py

experiments/
  README.md
  runs/                 # manifests only; generated data stays external
```

The Python-facing schema remains stable regardless of the archive implementation. Start with JSON plus compressed `.npz` for the tiny fixture and smoke dataset; make any later sharding format an implementation detail of `data.export` and `data.load`.

## How a clip is generated

One clip is produced by a pure, replayable pipeline:

1. **Derive random streams.** Build child generators from a master seed and record the seed tree. Do not use global random state.
2. **Sample a scene.** Sample object count, shape, size, pose, visible appearance, mass, friction, restitution, and initial zero velocity. Reject and resample overlaps or invalid placements.
3. **Build the physics space.** Create a Pymunk `Space` with fixed solver settings, zero gravity, static wall segments, and one dynamic body/shape per object. Store the mapping from stable object slot to body/shape.
4. **Render frame 0.** Render the initial scene before applying the action. Save RGB, per-object masks, selected-object mask, and initial centres.
5. **Apply the action.** Apply the configured force at the selected body's centre for the fixed duration. The force vector and action-map normalization are recorded exactly.
6. **Advance fixed substeps.** Call `space.step(dt)` at an internal timestep. At every substep, update contact state and collect collision onsets; at each output timestamp, snapshot poses, velocities if enabled, masks, and RGB.
7. **Finish labels.** Convert world positions to normalized coordinates, collapse substep events into recorded-frame intervals, and separate object-object from wall contacts.
8. **Write the sample.** Save arrays to `.npz` and replay metadata/config/seeds to JSON. Include schema, simulator, renderer, asset, and config hashes.
9. **Validate and optionally replay.** Reload the sample, reconstruct the space from metadata, regenerate it, and compare frames, masks, positions, and events within documented tolerances.

The first smoke command should generate one condition with several hidden-physics variants. Those variants must share the exact initial visible rendering, selected mask, and action while differing only in hidden mass/friction. This is the foundation of the uncertainty experiment.

## Hardware-aware data and model gates

Use measured gates rather than guessing a final dataset size:

- **A1 smoke:** 10–100 clips, one process, full replay checks.
- **A2 pilot:** approximately 1,000 clips, profile generation time, storage, label validity, and split checks.
- **A3 training pilot:** scale only after the compact 64×64 autoencoder/diffusion path reports peak VRAM and seconds/step on the RTX 4060.
- **Main dataset:** choose the count from the measured training budget; stream from disk and keep the same scene-group split for all model variants.

The 8 GB GPU is a model constraint, not a simulator constraint. If training is tight, reduce latent width, channel width, batch size, or clip length before compromising the simulator labels or leakage controls.

## Delivery phases

### Phase A0 — Contract and fixture

Maps to T01/T04.

Deliverables:

- resolved simulator/data decisions
- typed scene, object, physics, action, and rollout records
- one hand-written tiny scene fixture
- schema version and validation rules
- one example of a valid exported sample

Acceptance checks:

- a downstream consumer can load the fixture without importing the physics backend;
- tensor shapes, units, timestamps, and object-slot semantics are explicit;
- hidden physics appears only in oracle metadata/labels, never in model-conditioning fields;
- zero-force behavior and selected-object semantics are defined.

### Phase A1 — Package and CLI skeleton

Maps to T03 and the first part of T05.

Deliverables:

- importable `click2future` package
- configuration loading and resolved-config printing
- `generate-smoke` command
- `replay` command
- `validate-sample` command
- CPU device/environment check
- `requirements.txt` with pinned dependencies

Acceptance checks:

- a clean environment can install `requirements.txt` and import from `src`;
- each command has `--help` and deterministic seed arguments;
- the smoke command produces a small local artifact outside Git-tracked source directories.

### Phase A2 — Deterministic simulator and renderer

Maps to T05.

Deliverables:

- seeded scene generator for 2–4 objects
- bounded walls and supported shapes
- fixed-centre force application
- deterministic 64×64 renderer
- per-frame object masks and positions
- saved scene specification, resolved config, seeds, and version identifiers

Required tests:

- same seed/config/action produces identical metadata, masks, trajectories, and frames;
- changing only the action changes the rollout while preserving the initial scene;
- zero force keeps initially stationary objects stationary apart from defined numerical tolerance;
- reversing force reverses the selected object’s initial displacement direction;
- a controlled fixture produces a known object-object collision and wall contact.

### Phase A3 — Labels, replay, and sample export

Maps to T05/T06.

Deliverables:

- collision-onset extraction for object pairs
- separate wall-contact labels
- object validity and stable slot ordering
- `condition_id` and `scene_group_id` generation
- archive writer and reader
- replay comparison report with documented tolerances

Acceptance checks:

- collision persistence is recorded as one onset, not one event per frame;
- padded object slots are masked consistently;
- saved samples can be replayed from scene metadata, hidden physics, assets, action, and resolved config;
- replay differences are either bit-identical or within the documented tolerance;
- round-tripping an archive preserves schema fields and dtypes.

### Phase A4 — Dataset generation and leakage-safe splits

Maps to T06.

Deliverables:

- smoke dataset generator
- split manifest writer and hash
- train/validation/ID-test manifests
- shape-OOD, texture-OOD, force-OOD, and repeated-hidden-physics manifests
- split integrity and leakage checks
- generation/storage profiling record

Rules:

- split by `scene_group_id`, never by individual clip;
- keep all actions and hidden-physics variants of one visible scene together;
- keep rerenders and resolution variants in the same group;
- verify identical frame-0 inputs within each `condition_id`;
- verify hidden mass/friction are independent of visible appearance in generated metadata;
- freeze test manifests before model tuning.

### Phase A5 — Downstream handoff

Maps to T07/T08/T10/T17 dependencies.

Deliverables:

- documented Python API for dataset loading and batching
- a small fixed fixture consumed by evaluation and model smoke tests
- tracker/contact-estimator validation fixtures
- input normalization and frame-0 preservation rules
- example run manifest containing config, schema, split, seed, and hashes
- integration test from scene generation → archive → loader → metric input

Acceptance checks:

- model code can consume samples without knowing simulator internals;
- evaluation can distinguish simulator references from generated outputs;
- all downstream consumers use the same frame timing, coordinate system, and object identities;
- a clean checkout can regenerate the smoke fixture without external data.

## Stable Python-facing sample contract

The first implementation should expose the following logical fields, matching [ARCHITECTURE.md](ARCHITECTURE.md):

| Field | Shape/type | Notes |
|---|---|---|
| `frames` | `uint8 [T,H,W,3]` | RGB clip including frame 0 |
| `instance_masks` | `bool [T,N,H,W]` | Oracle masks |
| `object_valid` | `bool [N]` | Padding mask |
| `selected_object_id` | `int` | Stable frame-0 slot |
| `selected_mask` | `bool [1,H,W]` | Frame-0 selected mask |
| `force_xy` | `float32 [2]` | Documented world units |
| `action_map` | `float32 [2,H,W]` | Masked normalized force |
| `timestamps` | `float64 [T]` | Strictly increasing, starts at 0 |
| `positions` | `float32 [T,N,2]` | Normalized center coordinates |
| `collision_onset` | `bool [T-1,N,N]` | Object-object events |
| `wall_contact_onset` | `bool [T-1,N,4]` | Fixed wall order |
| `scene_spec` | object | Replayable initial state |
| `oracle_physics` | object | Labels/replay only |
| `seeds`, `sim_version`, `render_version` | metadata | Reproducibility |
| `config_hash`, `asset_hashes` | metadata | Provenance |

Batching into `[B,T,3,H,W]` and normalization should happen in the data loader, not in the simulator exporter. Keep the raw archive representation unambiguous.

## Definition of done

Workstream A is ready for model integration when all of the following are true:

- [ ] Contract decisions are recorded and versioned.
- [ ] The package imports in a clean CPU environment.
- [ ] A smoke rollout can be generated from a fixed seed.
- [ ] The rollout can be replayed from saved metadata.
- [ ] Frames, masks, positions, action, timestamps, and collision labels pass schema validation.
- [ ] Determinism, zero-force, force-reversal, collision, and wall-contact tests pass.
- [ ] Dataset splits are scene-group based and leakage checks pass.
- [ ] The loader round-trips a sample and produces documented tensors.
- [ ] A downstream smoke consumer can use the fixture without simulator-specific code.
- [ ] Generation commands, schema version, config hash, and artifact locations are documented.

## Recommended first PR

Keep the first PR deliberately small:

1. Add this plan and update the task links.
2. Add the package/module directories and `__init__.py` files.
3. Add typed contract records and schema validation.
4. Add one tiny fixture and round-trip test.
5. Add a `generate-smoke` placeholder that fails clearly until the backend is selected.
6. Add the initial dependency/configuration scaffolding through T03.

Do not generate the main dataset or commit a physics-engine choice until the contract gate is reviewed. The first meaningful technical milestone is a replayable, deterministic CPU fixture—not a large dataset.

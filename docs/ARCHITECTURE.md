# Architecture and data contract

The Workstream A simulator and dataset contract below is frozen as of 15
October 2026 for `scene-0.1`, `rollout-0.1`, and `dataset-0.1`. Incompatible
changes require a schema-version change and regenerated data. The later model,
auxiliary-head, and demo sections remain guidance for their owning workstreams.

## Frozen physics and action semantics

- The world is a 1 × 1 top-down plane with fixed walls and zero gravity. `x`
  increases right and `y` increases down, matching image coordinates. Because
  the world is 1 × 1, stored centres are already normalized to `[0,1]`.
- A scene contains 2–4 initially stationary circles or rectangles. Object IDs
  are assigned in sampling order and remain stable throughout the rollout.
- Pymunk 7.3.1 runs at 240 Hz with 10 solver iterations. The 32-frame pilot is
  recorded at 8 FPS, giving 30 physics steps per frame interval. Frame 0 is
  rendered before applying the action; timestamps run from 0 to 31/8 seconds.
- The selected object receives a centre-applied force for 0.25 seconds. Force
  angle is uniform over `[-pi,pi)` and magnitude is uniform over `[0.15,0.5)`
  in simulator mass·world-unit/s². Off-centre force and torque are out of scope.
- Hidden mass, friction, and restitution are sampled independently from visible
  geometry and appearance. Mass is uniform over `[0.5,2.0)`, friction over
  `[0.1,0.8)`, and restitution over `[0.0,0.4)`.
- Pymunk's Coulomb friction and elasticity rules are used. Object shapes carry
  the sampled coefficients; fixed wall segments keep Pymunk's frictionless,
  zero-elasticity defaults.
- Object-object and object-wall labels record contact onset. Begin events from
  all physics steps in `(timestamps[i], timestamps[i+1]]` are aggregated into
  label index `i`. Persistent contact is not repeated; separation followed by
  a new contact produces another onset.
- Wall order is fixed as `left`, `right`, `top`, `bottom`.

## Frozen sample schema

Each sample uses one compressed `.npz` array archive and one JSON metadata
file. Dataset membership and paths live in `dataset_manifest.json`; group-level
split membership lives in `splits.json`.

| Field | Shape/type | Storage and meaning |
|---|---|---|
| `sample_id` | string | JSON metadata and dataset manifest; unique archive stem |
| `scene_group_id` | string | Scene specification and manifest; shared visible scene |
| `condition_id` | string | Scene specification and manifest; same frame 0 and action |
| `split` | string | Dataset manifest; `train`, `validation`, or `test` |
| `scene_spec` | object | JSON metadata; geometry, initial state, action, seeds, and hidden physics |
| `frames` | uint8 `[T,H,W,3]` | NPZ; RGB clip including frame 0 |
| `instance_masks` | bool `[T,N,H,W]` | NPZ; oracle masks in object-ID order |
| `object_valid` | bool `[N]` | NPZ; all true before batching |
| `selected_object_id` | int | JSON metadata; stable selected-object slot |
| `selected_mask` | bool `[H,W]` | NPZ; selected object's frame-0 mask |
| `force_xy` | float32 `[2]` | NPZ; force in simulator units |
| `force_duration` | float | JSON metadata; action duration in seconds |
| `action_map` | float32 `[2,H,W]` | NPZ; selected mask multiplied by `fx` and `fy` |
| `timestamps` | float64 `[T]` | NPZ; strictly increasing seconds starting at zero |
| `positions` | float32 `[T,N,2]` | NPZ; object centres in `[0,1]` |
| `collision_onset` | bool `[T-1,N,N]` | NPZ; symmetric with a false diagonal |
| `wall_contact_onset` | bool `[T-1,N,4]` | NPZ; contacts in fixed wall order |
| hidden physics | object | `scene_spec.objects[].hidden_physics`; replay/oracle use only |
| versions and seeds | object | Scene and rollout metadata; deterministic reconstruction |

`dataset_manifest.json` stores the resolved simulator configuration and a hash
covering the manifest contents. `splits.json` stores its own split hash. There
are no external renderer assets in this version, so no asset hashes are needed.

`load_sample` converts frames to float32 `[T,3,H,W]` in `[0,1]` and expands the
selected mask to `[1,H,W]`. `load_batch` returns `[B,T,3,H,W]`, pads object
fields to the largest `N` in the batch, and marks real slots with
`object_valid [B,N]`.

Replay reconstructs the scene from JSON metadata and the recorded timing
configuration. `validate-sample --replay` defaults to absolute and relative
tolerances of `1e-6`; the pinned local environment currently reproduces every
stored field exactly.

## Leakage prevention and uncertainty groups

The hidden-physics model receives only frame 0, allowed frame-0 masks, and the
action. It never receives future masks, trajectories, collision labels, or
hidden physics as conditioning.

Each condition has four rollouts with identical frame 0, selected mask, and
action, varying only hidden mass, friction, and restitution. All variants of a
visible scene stay in one split through `scene_group_id`; clips are never split
independently. Visible, physics, and action random streams use separate recorded
seeds.

The ID split is 80% train, 10% validation, and 10% test by scene group using a
recorded split seed. Shape-, texture-, and force-OOD suites require separate
manifests and do not alter this frozen ID split. Rerenders or later resolution
variants of one scene must retain its group and split.

## Model interfaces

```text
encode(frames) -> latents
decode(latents) -> frames
build_condition(first_frame, selected_mask, force_xy) -> condition
denoise(noisy_future_latents, diffusion_step, condition) -> predicted_noise, optional_aux
sample(condition, seeds=[...], num_future_frames=T-1) -> videos
evaluate(videos, references, metadata) -> metric_records
```

The frame autoencoder should be trained or selected once, then frozen and
shared across diffusion ablations. Fit its weights and latent normalization
statistics using training data only. Use validation data for checkpoint
selection; evaluate test/OOD reconstruction only after selection is fixed.

For the conditional VAE, define a training posterior `q(z | future, condition)`,
a prior `p(z | condition)` or fixed standard-normal prior, and a decoder
`p(future | z, condition)`. At inference, sample from the prior without
supplying the future clip. Record the prior choice and KL weight in the run
configuration. See [the model references](REFERENCES.md#models).

## Proposed action module

1. Encode frame 0 into a spatial feature map.
2. Downsample the selected-object mask and pool a selected-object feature.
3. Fuse that feature with an embedding of the force vector.
4. Encode the spatial action map and inject the result into denoiser blocks.

E2 receives the selected mask and numeric force but not the spatial action path,
so the E2-to-E3 comparison isolates the spatial module.

## Auxiliary heads

Predict object-centre trajectories and object-pair collision probabilities from
denoiser features. Use the stable simulator object slots and `object_valid` for
padding. Oracle future masks, trajectories, collisions, and hidden physics are
training labels or evaluation references, never inference-time inputs.

```text
L = L_diffusion + lambda_traj * L_trajectory + lambda_collision * L_collision
```

Exclude diagonal and duplicate collision pairs, handle class imbalance, and
tune loss weights only on validation data. Image-derived evaluation must still
check whether generated videos follow predicted trajectories.

## Demo boundary

Preset scenes with compatible metadata support exact simulator replay. Uploaded
images without scene metadata support generation but not simulator-ground-truth
comparison.

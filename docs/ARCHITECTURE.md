# Architecture and data contract

This is a proposed interface to finalize in T04 before independent implementation.
An untrained pixel-space denoiser prototype exists; the data contract and end-to-end
training/sampling path are not implemented. See [the 14 October build proposal](BUILD_PLAN.md)
for the pixel pilot, equal-capacity action comparison, and optional latent path.

## Physics and action semantics

- Use a 2D top-down rigid-body world with fixed walls, initially stationary objects, and a deterministic seeded renderer.
- Coordinates are normalized to [0, 1], with x increasing right and y increasing down. Store the world-to-image transform and physical units explicitly.
- Frame 0 is rendered before the action. A force vector `(fx, fy)` acts at the selected object's centre for a fixed documented duration. The click selects the object; off-centre torque is outside the core scope.
- The physics timestep is independent of video fps. Record substeps, force duration, friction law, restitution, and collision tolerances.
- The UI arrow maps to the same force normalization as training. Define `magnitude = sqrt(fx² + fy²)`; derive direction from the vector. Handle zero force without dividing by zero.
- Randomize hidden mass and friction independently of visible shape, size, colour, and texture. Store them as oracle metadata, never in hidden-physics model inputs.
- Define object-object collision events from contact onset during each interval between recorded frames. Store object-wall contacts separately.

## Sample schema

An archive plus JSON metadata is sufficient for the first implementation. Choose the storage library in T03; keep tensor semantics stable.

| Field | Proposed shape/type | Meaning |
|---|---|---|
| `sample_id` | string | Unique rollout ID |
| `scene_group_id` | string | Visible scene identity, shared across actions and physics variants |
| `condition_id` | string | Same initial rendering, selected object, and action |
| `split` | enum | Train, validation, ID test, or named OOD test |
| `scene_spec` | object | Initial geometry, poses, velocities, walls, textures, and renderer settings for replay |
| `timestamps` | float64 `[T]` | Seconds from frame 0; strictly increasing, starting at zero |
| `frames` | uint8 `[T,H,W,3]` | RGB clip including frame 0 |
| `instance_masks` | bool `[T,N,H,W]` | Oracle per-object segmentation |
| `object_valid` | bool `[N]` | Valid object slots if padded |
| `selected_object_id` | int | Stable selected-object slot |
| `selected_mask` | bool `[1,H,W]` | Selected object's frame-0 mask |
| `force_xy` | float32 `[2]` | Force vector in documented world units |
| `force_duration` | float | Fixed action duration in seconds |
| `action_map` | float32 `[2,H,W]` | Selected mask multiplied by normalized fx and fy |
| `positions` | float32 `[T,N,2]` | Oracle object-centre coordinates in [0,1] |
| `collision_onset` | bool `[T-1,N,N]` | Symmetric object-object event matrix; diagonal false |
| `wall_contact_onset` | bool `[T-1,N,4]` | Separate wall-contact events |
| `oracle_physics` | object | Mass, friction, restitution; training labels/replay only |
| `seeds`, `sim_version`, `render_version` | metadata | Reproducibility and replay |
| `config_hash`, `asset_hashes` | metadata | Resolved simulator config and any texture/shape asset checksums |

When batching, convert images to float `[B,T,3,H,W]` using one documented normalization. Predict `T-1` future frames; retain the supplied frame 0 exactly in displayed output.

For 16 frames at 8 fps, use timestamps 0, 1/8, ..., 15/8 seconds. Collision index `i` describes the interval `(timestamps[i], timestamps[i+1]]`. A contact that persists across frames is one onset, not a new event each frame. Start objects without overlap and record any permitted initial contact separately. Fix the wall order in the contract, for example left, right, top, bottom.

Replay must load the scene specification, hidden physics, assets, action, and resolved simulator config. Verify a saved sample can be reconstructed in the pinned environment, with a documented tolerance if the physics engine does not reproduce results bit for bit.

## Leakage prevention and uncertainty groups

The hidden-physics model receives only frame 0, allowed frame-0 masks, and action. It never receives future masks, trajectories, collision labels, or oracle physics as conditioning.

For uncertainty evaluation, generate multiple rollouts with **identical frame 0, selected mask, and action**, varying only hidden mass/friction. Keep all actions and physics variants of a visible scene in the same split using `scene_group_id`. Do not split clips independently. Verify equality of initial rendered inputs within each `condition_id`.

Start with a small smoke dataset and size the main dataset after profiling. Persist the split manifest and its hash. Reserve disjoint shape families, texture families, and a force range for separate OOD tests; keep the other factors in distribution for each test.

Keep rerenders, augmentations, and different resolutions of the same base scene in the same scene group and split.

## Model interfaces

```text
encode(frames) -> latents
decode(latents) -> frames
build_condition(first_frame, selected_mask, force_xy) -> condition
denoise(noisy_future_latents, diffusion_step, condition) -> predicted_noise, optional_aux
sample(condition, seeds=[...], num_future_frames=T-1) -> videos
evaluate(videos, references, metadata) -> metric_records
```

For the proposed pixel pilot, encoding and decoding are identity operations. If the
latent path is selected, train/select the frame autoencoder once, then freeze and share
it across diffusion ablations. Fit weights and latent normalization on training data
only. Use validation for checkpoint selection and inspect test/OOD reconstruction after
freezing it. Record pretrained-weight provenance and training seeds. Require object
geometry and contact/trajectory preservation before accepting compression.

For the conditional VAE, define a training posterior `q(z | future, condition)`, a prior `p(z | condition)` or fixed standard-normal prior, and a decoder `p(future | z, condition)`. Train with a reconstruction loss and a weighted KL term. At inference, sample from the prior without supplying the future clip. Record the prior choice and KL weight in the run config. See [the model references](REFERENCES.md#models).

## Proposed action module

1. Encode frame 0 into a spatial feature map.
2. Downsample the selected-object mask and use masked pooling to obtain a selected-object feature.
3. Fuse that object feature with an embedding of the force vector.
4. Encode the spatial action map and inject the resulting spatial features and object/action embedding into denoiser blocks.

Select a small conditioning mechanism (for example feature concatenation plus feature modulation) during the pilot. Freeze the mechanism before the main ablation run. E2 receives the same selected mask and numeric force, but not this spatial action path, so a gain is not explained merely by identifying the target object.

## Auxiliary heads

Predict object-centre trajectories and object-pair collision probabilities from denoiser features. Use deterministic frame-0 object-slot ordering (e.g. centroid order); do not order using future ground truth. If all-object masks are used for pooling, provide the same frame-0 object extraction to every variant and document that additional input.

Any extra inference-time object extraction must use only the first-frame pixels. Oracle all-object masks may serve as training labels or evaluation references, but cannot be an extra input available only to E4. Keep the selected-object mask available to E0/E2/E3/E4 as specified in the comparison table. The E1 generator receives only the first frame.

Proposed objective:

```text
L = L_diffusion + lambda_traj * L_trajectory + lambda_collision * L_collision
```

Mask padding, exclude diagonal/duplicate collision pairs, and handle collision imbalance. Specify which diffusion timesteps supervise the heads and whether heads predict clean trajectories from noisy features. Tune weights only on validation data. The heads support learning; image-derived evaluation must still check whether the actual generated videos follow those trajectories.

## Demo boundary

Preset scenes carry metadata and support exact simulator replay. Synthetic image uploads support generation after selecting or providing a mask; exact simulator comparison is disabled unless compatible scene metadata is also supplied. Friction controls the simulator in hidden-physics mode. A future observed-friction model would need its own training input, config, checkpoint, and clearly labelled mode.

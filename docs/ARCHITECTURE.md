# Architecture and data contract

The Workstream A simulator and dataset contract below is frozen as of 15
October 2026 for `scene-0.1`, `rollout-0.1`, and `dataset-0.1`. Incompatible
changes require a schema-version change and regenerated data. The later model,
auxiliary-head, and demo sections remain guidance for their owning workstreams.
A pixel denoiser, dataset trainer, DDIM sampler, CVAE baseline and
extracted-trajectory/contact metrics are implemented. The 16 October decision
adopts the pixel path; see [BUILD_PLAN.md](BUILD_PLAN.md), [diffusion](TRAINING.md)
and [CVAE training](MODEL_TRAINING.md) for evidence gates.

## Frozen physics and action semantics

- The world is a 1 × 1 top-down plane with fixed walls and zero gravity. `x`
  increases right and `y` increases down, matching image coordinates. Because
  the world is 1 × 1, stored centres are already normalized to `[0,1]`.
- A scene contains 2-4 initially stationary circles or rectangles. Object IDs
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
denoise(noisy_future_pixels, diffusion_step, condition) -> predicted_noise, optional_aux
sample(condition, seeds=[...], num_future_frames=T-1) -> videos
evaluate(videos, references, metadata) -> metric_records
```

For the adopted pixel path, encoding and decoding are identity operations. If the
latent path is selected, train/select the frame autoencoder once, then freeze and share
it across diffusion ablations. Fit weights and latent normalization on training data
only. Use validation for checkpoint selection and inspect test/OOD reconstruction after
freezing it. Record pretrained-weight provenance and training seeds. Require object
geometry and contact/trajectory preservation before accepting compression.

For the conditional VAE, define a training posterior `q(z | future, condition)`,
a prior `p(z | condition)` or fixed standard-normal prior, and a decoder
`p(future | z, condition)`. At inference, sample from the prior without
supplying the future clip. Record the prior choice and KL weight in the run
configuration. See [the model references](REFERENCES.md#models).

## Implemented conditioning comparison

Both E2 and E3 concatenate noisy future RGB with repeated frame-0 RGB,
the selected mask and two force channels. Timestep and numeric force embeddings
modulate residual blocks. E2 broadcasts the force vector across the image; E3
multiplies those channels by the selected mask. The same U-Net predicts epsilon.

E2/E3 have identical information, parameter counts and topology. The experiment
isolates force-field localization. Pooled object features are an optional later
extension, not part of this implemented model.

## Auxiliary heads

The prototype predicts object-centre trajectories and object-pair collision logits
from denoiser features. The implemented label adapter sorts frame-0 mask centres by x,
then y, giving heads a visual slot order. Simulator sampling-order IDs cannot be inferred
from RGB. One fixed permutation covers the full clip and both collision axes; it is
reversed for head evaluation. Padding is excluded with valid masks. Oracle future
trajectories and collisions are training labels or evaluation references, never inputs.
See [loss implementation and integration](AUXILIARY_LOSSES.md).

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

The native PNG endpoint accepts pixels only (64 x 64, opaque, at most 64 KiB).
The existing observable-component parser derives two masks in centroid order;
neither a scene recipe nor hidden properties can be supplied. The in-memory input
contains one observed frame and an explicit 32-frame output length, without fake
future arrays. Reference generation is disabled for these inputs. The same model
adapter, four ordered sampling seeds and observable replay-cache identity are
used for uploaded and preset images. At most eight unique uploads remain open.

### Separate photograph extension

`photo_server` accepts ordinary photographs through a separate interface and
durable request queue. `scripts/photo_remote.py` synchronizes that queue over
verified SSH to an allocated `cluster/photo_worker.py` process. The released
external Force Prompting/CogVideoX model produces four ordered RGB videos;
no simulator metadata or ground-truth masks are supplied. Model identity binds
checkpoint, source/runtime, sampler profile and video contract. Input, requests,
all raw samples and presentation MP4 hashes are retained. Cached replay is
explicit and available offline; fresh inference needs a live allocated worker.
This extension is separate from the original models and evaluation evidence.
See [photo setup and limitations](PHOTO_DEMO.md).

# Auxiliary trajectory and collision losses

T12 has target preparation, fixed object matching, masked losses, class-balance estimation,
and CPU gradient checks. The dataset diffusion trainer and sampler are still pending.
The starting configuration is [auxiliary.yaml](../configs/training/auxiliary.yaml).
Its weights and timestep cutoff are untuned proposals, to be checked on validation data.

## Object identity and target shapes

Simulator IDs follow random sampling order. That order cannot be inferred from an image,
so supervising head slot 0 as simulator ID 0 would create ambiguous targets. The label
adapter instead sorts valid frame-0 mask centres by x, then y. Padding comes last. It
rejects overlapping/empty masks and identical centres. It never rematches objects using
future positions, collisions, or the model's predictions.

`build_auxiliary_targets` takes the loader's positions `[B,T,N,2]`, boolean collision
onsets `[B,T-1,N,N]`, `object_valid [B,N]`, and **only frame-0** instance masks `[B,N,H,W]`.
After NumPy-to-Torch conversion, pass tensors on the same device. It returns:

| Field | Shape | Meaning |
|---|---|---|
| `positions` | `[B,T-1,M,2]` | Future centres in normalized image coordinates |
| `collision_onset` | `[B,T-1,M,M]` | Symmetric onset labels in canonical order |
| `position_valid` | `[B,T-1,M]` | Real object and available future frame |
| `pair_valid` | `[B,T-1,M,M]` | Real upper-triangle pair with both interval endpoints available |
| `canonical_to_scene` | `[B,M]` | Original simulator IDs, or -1 for padding |

M is the model's `max_objects`. The same permutation applies to every future frame and
both collision axes. Changing future trajectories cannot change the mapping. Frame 0 is
removed from position targets; collision index 0 still means the frame-0-to-frame-1 interval.
Optional boolean `future_valid [B,T-1]` masks unavailable labels. Valid reference positions
must be finite and in [0,1]; ignored padding/missing positions are cleared before loss use.

Masks and future labels are used to align supervision, not as additional model inputs.
The model still receives only noisy future frames, diffusion timestep, first-frame RGB,
selected-object mask, and numeric force. It must learn the canonical visual slot order.
The dataset's saved simulator IDs and tensor schema stay unchanged.

## Loss and timestep policy

The trainer adds `aux.total` to its diffusion objective:

```python
from click2future.training import AuxiliaryLossConfig, auxiliary_loss, build_auxiliary_targets

# batch contains Torch tensors converted from data.load_batch, on the model device.
targets = build_auxiliary_targets(
    batch["positions"], batch["collision_onset"], batch["object_valid"],
    batch["instance_masks"][:, 0], max_objects=model.max_objects,
)
output = model(noisy_future, timesteps, first_frame, selected_mask, normalized_force)
aux = auxiliary_loss(
    output, targets, timesteps,
    num_diffusion_steps=num_diffusion_steps,
    config=AuxiliaryLossConfig(),
)
loss = diffusion_loss + aux.total
loss.backward()
```

The trainer owns the noise schedule, timestep sampling, input/force normalization, optimizer,
and checkpoints. These components do not implement a diffusion trainer or sampler.

The trajectory term uses [Smooth L1](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)
on normalized x/y coordinates, with beta 0.05. The collision term uses
[BCE with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
on valid unordered object pairs. Diagonal, duplicate lower-triangle, and padded entries
are excluded. Position and collision loss calculations/reductions use float32, including
when head outputs use reduced precision. Each term first averages valid entries within
each clip, then averages eligible clips. A clip with three objects does not outweigh one
with two solely because it has more labels.

Default weights are trajectory 1.0 and collision 0.1. Both terms support a zero-weight
ablation. Empty valid masks return a differentiable zero; ignored entries are removed
before arithmetic, so NaNs in padding cannot contaminate loss or gradients.

By default, clean future labels supervise only timesteps at or below
`floor(0.25 * (num_diffusion_steps - 1))`. For 1000 steps, that is 0-249 inclusive.
Higher-noise clips receive no auxiliary gradient. This timestep fraction is a proxy for
noise level and depends on the eventual schedule; validate it before main experiments.
The cutoff, component counts, config, raw terms and weighted total are available through
`aux.record()` for training logs.

## Collision class imbalance

Use `collision_balance(training_target_batches)` once on the training split's label counts.
It counts each valid unordered interval/pair once, estimates negative/positive count ratio,
and clamps it to [1,20] by default. Record the training manifest, counts, and resulting
weight, then set `collision_positive_weight` in the run config. Do not recompute it from
each minibatch or from validation/test labels. The default weight remains 1 until this
training-set estimate is available.

Empty pairs, no positives, or no negatives use weight 1 with an explicit status. An
all-negative dataset still cannot demonstrate useful collision learning. Weighted BCE is
normalized by the valid pair count, not by the sum of positive/negative weights.

## Head evaluation and inference

For inference, compute `canonical_object_order` from the shared frame-0 identity masks;
no future labels or target builder are required. `to_scene_order` maps future positions
and both collision axes back to simulator IDs, leaving padded output slots as NaN. Use
that fixed mapping for the entire clip. Do not rematch using future ground truth to
improve scores.

The existing [metric exporter](METRICS.md#run) accepts `--source auxiliary_head`. Prepend
frame-0 coordinates to future position predictions for its `[K,T,N,2]` interface, and use
a validation-selected threshold on collision logits. Record the checkpoint, denoiser
step/noise level, threshold, and mapping version in the run manifest. Head scores must
remain separate from RGB tracking metrics; correct heads do not prove correct videos.
The sampler must define which denoiser call supplies the reported head predictions.

## Reproduce the correctness checks

```bash
python -m click2future.training.check_auxiliary \
  --output outputs/auxiliary-loss-checks.json
python -m pytest tests/test_auxiliary_losses.py -q
```

The checked-in [record](results/auxiliary-loss-checks.json) uses two seeded simulator
clips with two/three objects, four frames at 64px, padding, and a known collision.
It runs backward through the real base-width-8 denoiser and verifies gradients in the
trajectory head, collision head, and bottleneck. No generator optimizer update occurs.
A separate check optimizes explicit prediction tensors for 60 steps; loss fell from
0.19974 to 0.00143. This checks the objective's mechanics, not learned video quality.
Fixture class counts are not a production training-set estimate.

Tests cover simulator-ID permutation, crossing futures without rematching, both collision
axes, padding and missing frames, timestep boundaries, component ablations, class-weight
caps/degenerate cases, mixed-precision head outputs, and finite/nonzero real-model gradients.
E4 dataset training, tiny-batch model overfitting, checkpoint/sampling checks, and measured
comparisons remain to be completed with B.

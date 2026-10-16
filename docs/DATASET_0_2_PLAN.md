# Dataset 0.2 Plan

Dataset 0.2 adds controlled action comparisons to Workstream A while keeping
the existing dataset-0.1 generator and its output contract unchanged.

## Dataset unit

One scene group has the same frame-0 geometry, appearance, and object poses
across every clip. The action specifies which object is selected. It contains:

| Action | Hidden-physics variants | Purpose |
|---|---:|---|
| `zero` | 1 | Check that an initially stationary scene stays stationary |
| `forward` | 4 | Measure motion along a nonzero force without contact |
| `reverse` | 4 | Measure motion under the exact opposite force without contact |
| `collision_0_to_1` | 4 | Push object 0 into object 1 in a glancing collision |
| `collision_1_to_0` | 4 | Push object 1 back into object 0 in a glancing collision |
| `wall` | 4 | Learn boundary contact and its effect on future frames |

This gives 21 clips per scene group. Dataset 0.2 uses exactly two objects so
the controlled contact geometry stays simple and reliable.

## Controlled construction

- Object 0 starts near the centre of a unit tabletop.
- The other object is placed away from the forward/reverse path.
- `forward` and `reverse` use equal and opposite forces and must remain free of
  object and wall contacts for the full clip.
- The two collision actions select opposite objects and push them toward each
  other. Their shared layout alternates left/right offsets across scene seeds,
  producing glancing contacts from both sides of the tabletop dataset.
- `wall` pushes the selected object into a boundary without touching the other
  object and must produce at least one wall contact.
- `zero` uses a zero force and must contain no motion or contacts.
- The four hidden-physics profiles vary mass, friction, and restitution. A
  profile index means the same hidden physics in `forward`, `reverse`, and
  both collision directions, and `wall`, which makes cross-action comparisons
  controlled.
- The zero clip uses profile 0 as its canonical hidden-physics state; extra
  zero variants would be duplicates because there is no motion or contact.

The action is visible to the model through `force_xy` and `action_map`. Hidden
mass, friction, and restitution remain available only as oracle metadata.

The default physical force magnitudes are 0, 0.15 for forward/reverse, 0.25 for
object collisions, and 1.1 for wall contact. The dataset-0.2 training config
declares `force_limit: 1.1`; the dataset-0.1 default remains 0.5. Both use the
same force normalization divisor of 0.5. Training and checkpoint-driven sampling
accept all six action families without dropping wall clips. See [training](TRAINING.md).

## Identity and splits

- `scene_group_id` identifies all 21 clips with the same visible frame 0.
- `condition_id` identifies one action within a scene group.
- `action_variant` identifies zero, forward, reverse, either collision
  direction, or wall contact.
- `physics_variant` is `0` through `3`; it is `0` for the canonical zero clip.
- Every clip from a scene group is assigned to the same train, validation, or
  test split. No visible scene can leak across splits.

The recommended full dataset has 200 scene groups and 4,200 clips. With the
default 80/10/10 grouped split this gives 160/20/20 scene groups and
3,360/420/420 clips. A small pilot is generated and validated before the full
run.

## Output contract

Individual clips keep the existing `rollout-0.1` array and metadata contract,
so current loaders and replay tools continue to work. The dataset manifest is
`dataset-0.2` and adds action/physics labels plus observed contact summaries.
Generated data remains under `outputs/` and is not committed.

## Validation gates

Generation fails if any scene group violates these conditions:

1. Every clip has the same visible frame 0 and the expected selected object.
2. The zero clip has no measurable displacement or contact.
3. Forward and reverse move in their requested directions and have no object
   or wall contacts.
4. Their forces are exact negatives of each other.
5. Both collision directions record object-object collision without wall contact.
6. Every wall variant records wall contact without an object-object collision.
7. A physics profile is identical across all nonzero actions.
8. All clips pass the existing rollout validator and reproduce by exact replay.
9. All 21 clips from a scene group share one split.

## Implementation steps

1. Add a deterministic controlled-scene builder for the six actions.
2. Add the dataset-0.2 generator, manifest records, and grouped splits.
3. Add dataset-wide validation and action/contact summaries.
4. Add a simulator config, CLI commands, documentation, and focused tests.
5. Generate a small pilot, inspect its event counts and frames, and run replay.
6. Generate the full 200-group dataset only after the pilot passes.

## Non-goals

Dataset 0.2 does not add surface drag, gravity, 3D motion, stacking,
photorealism, or a new model architecture. Friction remains a contact property:
it affects object-object collision response, not free sliding through empty
space.

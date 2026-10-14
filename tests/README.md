# Verification

The current tests cover the simulator/data scaffold:

- deterministic scene generation and hidden-physics variants;
- force direction, zero-force behavior, and contact events;
- deterministic RGB and instance-mask rendering;
- rollout shapes, timestamps, action maps, and oracle labels;
- JSON/NPZ round trips and schema validation.
- saved-rollout replay and mismatch detection.
- model-facing channel-first sample and padded batch loaders.

Model, evaluation, split, replay, and end-to-end CLI tests will be added as
those layers are implemented.

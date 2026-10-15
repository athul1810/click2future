# Verification

The current tests cover the simulator/data scaffold:

- deterministic scene generation and hidden-physics variants;
- force direction/reversal, zero-force behavior, and known contact events;
- deterministic RGB and instance-mask rendering;
- rollout shapes, timestamps, action maps, and oracle labels;
- JSON/NPZ round trips and schema validation.
- saved-rollout replay and mismatch detection.
- model-facing channel-first sample and padded batch loaders.
- the fixed scene fixture and end-to-end generate/validate CLI flow.

Model and evaluation tests will be added by their owning workstreams.

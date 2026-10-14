# Planned verification

Add checks alongside implementation for behavior that matters:

- Deterministic simulator replay, force direction, collision events, and zero-force behavior.
- Tensor shapes, units, valid-object masks, split disjointness, and absence of hidden-physics leakage.
- Metrics on known trajectories, contact events, identical videos, and deliberately corrupted videos.
- Autoencoder reconstruction and model forward/backward/sample smoke runs.
- Resume behavior and one complete demo-to-sampler path.

`test_environment.py` checks the CPU diagnostic/report and the failure status for an
unavailable CUDA device. Run `python -m pytest -q` using the environment in
[development setup](../docs/SETUP.md). Model and simulator tests are not available yet.

# Planned configurations

Development configurations use YAML; `development.yaml` records artifact-directory
conventions only. It is a template, not yet consumed by a simulator or training command.
`pilot.yaml` drives the synthetic profiler's dimensions and documents the proposed
architecture. Its simulator/training values are proposals, not a working trainer config.
T04 will finalize domain settings and T13 will freeze the main experiments. Planned groups:

- Simulator: world units, timestep, force duration, object/appearance/physics distributions.
- Dataset: scene groups, split seeds, held-out families/ranges, resolution and clip length.
- Model: shared autoencoder, denoiser sizes, conditioning mode, auxiliary loss weights.
- Training: optimizer, steps, batch size, precision, checkpoint policy, seeds.
- Evaluation: split manifest, sampler, sample count, metric options.

Track complete resolved configs with each run. Values in the planning documents are proposals until validated.

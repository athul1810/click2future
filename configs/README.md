# Configurations

TOML is the configuration format. `simulator/smoke.toml` is the short smoke
configuration, while `simulator/pilot.toml` is the 32-frame pilot setting.
Later tasks will add the full simulator, dataset, model, training, and
evaluation settings. T13 will freeze the main experiments.

- Simulator: world units, timestep, force duration, object/appearance/physics distributions.
- Dataset: scene groups, split seeds, held-out families/ranges, resolution and clip length.
- Model: shared autoencoder, denoiser sizes, conditioning mode, auxiliary loss weights.
- Training: optimizer, steps, batch size, precision, checkpoint policy, seeds.
- Evaluation: split manifest, sampler, sample count, metric options.

Track complete resolved configs with each run. Values in the planning documents are proposals until validated.

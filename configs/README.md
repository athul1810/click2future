# Planned configurations

T03/T04 will establish the configuration format; T13 will freeze the main experiments. Planned groups:

- Simulator: world units, timestep, force duration, object/appearance/physics distributions.
- Dataset: scene groups, split seeds, held-out families/ranges, resolution and clip length.
- Model: shared autoencoder, denoiser sizes, conditioning mode, auxiliary loss weights.
- Training: optimizer, steps, batch size, precision, checkpoint policy, seeds.
- Evaluation: split manifest, sampler, sample count, metric options.

Track complete resolved configs with each run. Values in the planning documents are proposals until validated.

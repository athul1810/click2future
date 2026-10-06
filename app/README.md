# Planned interactive demo

T17 owns the initial implementation. It should load a preset scene or a supported synthetic image, select an object, map a drag arrow to the agreed force vector, display four sampled futures, and show a simulator reference when scene metadata exists.

Use the same preprocessing and sampler as evaluation. Display checkpoint identity, seed controls, loading/error states, and clearly label generated versus simulated output. Friction changes simulator ground truth for the core hidden-physics model; direct model friction control requires the optional conditioned model.

Start with the lightest suitable interface after the model/data contract is stable. No app is implemented yet.

# Planned interactive demo

The [demo specification](DEMO_SPEC.md) defines the interaction, real-model integration,
comparison/uncertainty story, optional goal search, acceptance checks, and presentation
fallback. Its delivery targets are in the [16 October execution plan](../docs/EXECUTION_PLAN_2026-10-16.md).
The PPT presentation is on 14 November; demo ownership remains pending.

T17 owns the initial implementation. It should load a preset scene or a supported synthetic image, select an object, map a drag arrow to the agreed force vector, display four sampled futures, and show a simulator reference when scene metadata exists.

Use the same preprocessing and sampler as evaluation. Display checkpoint identity, seed controls, loading/error states, and clearly label generated versus simulated output. Friction changes simulator ground truth for the core hidden-physics model; direct model friction control requires the optional conditioned model.

Start with the lightest suitable interface after the model/data contract is stable. No app is implemented yet.

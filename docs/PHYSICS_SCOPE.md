# Contact friction and surface drag

Decision for the first trained comparison, 16 October: retain `scene-0.1`
and its contact-only friction. Preserve `dataset-0.1`; training can start now.

`hidden_physics.friction` is assigned to Pymunk object shapes. It affects
object-object contact, not an object's motion across an invisible tabletop.
Gravity is zero and space damping is one. A contact-free object coasts after
the 0.25 s force ends. Changing friction alone cannot change that trajectory.
Mass still changes acceleration. Friction/restitution need suitable contact
cases to be identifiable; random free-motion clips cannot establish that the
model learned either coefficient.

Walls currently have friction and elasticity zero. Pymunk multiplies the two
shapes' coefficients, so object-wall contacts are frictionless with zero
restitution even when the object's coefficients are nonzero. This is already
the frozen architecture contract, not a newly introduced physics change.
See the [Pymunk contact rules](https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.friction).

## Handoff to Sanjay

1. Review the regression fixtures in `tests/test_friction_contract.py` and
   confirm the report/demo describes contact friction rather than sliding drag.
2. For the next dataset version, provide separate free-motion, glancing
   object-contact, head-on contact and wall-contact strata. Include zero,
   reversed and multiple nonzero actions for each visible scene, keeping every
   action/physics variant of a scene in the same split. Report event counts.
3. If surface slowdown is needed for the demo, propose it as a separate version:
   a named drag law and units, independent coefficient, wall coefficients,
   force-off stopping behavior, and tolerances at 120/240/480 Hz. Coulomb drag
   needs an explicit clamp at zero speed; a multiplicative per-step factor
   without time scaling changes the physics with timestep. Compare trajectories
   and stopping distance before selecting ranges. Do not relabel the existing
   contact-friction field or overwrite old samples.

The additional drag version is optional. The core experiment compares learned
video models under the current declared physics. Neither adding drag nor changing
wall restitution is required to unblock the trainer.

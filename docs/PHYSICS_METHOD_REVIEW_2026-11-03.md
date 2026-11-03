# Physics method selection, 3 November 2026

The goal is reliable two-object tabletop futures from one resting 64px image,
a selected mask and a centre-applied force. Appearance is synthetic, gravity is
zero, the pulse lasts 0.25 seconds and hidden mass/friction/restitution are
independent of appearance. A newer general video model is not automatically a
better match. No paper establishes which method wins on our exact protocol.

## Primary-source review

| Method | Evidence inspected | Decision for this project |
|---|---|---|
| [PhysGen, ECCV 2024](https://arxiv.org/pdf/2409.18964), [official code](https://github.com/stevenlsw/physgen) | Full dynamics/rendering method and released simulation instructions: image-derived primitives, explicit rigid-body state evolution, compositing and optional diffusion refinement. | Closest deployable mechanism for this 2D task. Adapt its explicit physics stage and existing observed-pixel rendering as a labelled baseline. Our implementation does not reproduce its VLM perception, relighting or SEINE refinement. |
| [IMPACT, 1 November 2026](https://arxiv.org/html/2611.01372v1) | Sections III and IV: event-driven free flight and impacts, bounded physical calibration, trajectory/dense guidance; controlled sphere-plane evaluation. | Supports explicit continuous/discrete state evolution. Its gravity-driven single-sphere setting and reported rankings do not transfer to our two-object push task. We borrow the physical-state principle, not its weights or numerical results. |
| [RiCo, October 2026](https://arxiv.org/html/2610.12333v1) | Full state interface, local contact neighborhoods, anchor prediction, autoregressive rollout and training appendix. | Strong candidate for a learned dynamics redesign. It consumes two states, surface geometry and physical properties. Adapting it requires new state-transition training and rollout validation, rather than adding another attention block to a whole-clip decoder. |
| [Graph network contact study, CoRL](https://proceedings.mlr.press/v205/allen23a/allen23a.pdf) | Forward-dynamics representation, history/velocity features and contact experiments. | Shows learned graph dynamics can model contacts; our failures do not prove neural simulation impossible. Its mesh-state supervision and initialization differ from one-image video generation. |
| [Goal Force, CVPR 2026](https://github.com/brown-palm/goal-force), [official project](https://goal-force.github.io/) | Released Wan2.2/ControlNet workflow, inputs and compute requirements. | Relevant external force-conditioned generator. Published training uses four A100s for two days; it is not the bounded single-A40 recipe already measured here. Pretrained inference would need a separate force-unit, resolution and memory adaptation before a fair comparison. |
| [PhysCtrl](https://arxiv.org/html/2509.20358v2) | Dynamics representation, losses and implementation appendix. | Rich 3D/material trajectory generator, with 550K animations and multi-GPU published training. It changes data, geometry, conditioning and compute substantially; not a drop-in repair of our two-object 64px predictor. |
| [Few-Shot Neural Differentiable Simulator, May 2026](https://arxiv.org/html/2603.06218v2) | Contact identification, data scaling and GNN simulation method. | Supports calibration plus structured state prediction. We do not have real contact measurements to reproduce its real-to-sim identification claim. |
| [PhysRepair, October 2026](https://arxiv.org/html/2610.23677v1) | Generate-diagnose-repair method and explicit local simulator step. | Reinforces evidence-driven diagnosis. Its agent/perception/regeneration stack adds complexity; our exact synthetic labels make direct mechanics checks more appropriate than a VLM verdict. |

These are mechanism and feasibility comparisons, not reproduced external model
rankings. The running D020 experiment remains a candidate; its altered training
margins do not have a literature-backed optimality guarantee. Its validation
rest failure is fixed, but penetration and generalization must still be measured.

## Physics that can actually be checked

For a contact-free, initially stationary object of mass m under force F for a
pulse duration tau:

- For t <= tau, velocity is F*t/m and displacement is F*t*t/(2*m).
- For t > tau, velocity is F*tau/m and displacement is F*tau*(t-tau/2)/m.
- Zero force gives exact rest. Reversing force reverses displacement before
  contact; doubling mass halves it. These are deductions from Newton's law,
  not empirical assumptions about a neural network.

For an isolated pair, collision impulses are equal and opposite, so total
linear momentum changes only by the applied external impulse. Friction and
restitution below one dissipate kinetic energy. Therefore forcing total energy
conservation through an ordinary conservative Hamiltonian model would encode
the wrong contact physics for this dataset.

A numerical integrator only approximates continuous motion. Tests must check
its discrete update and convergence as the step is halved. Our baseline uses
240 Hz and records a 0.1-native-pixel solver overlap allowance in world units.
A half-pixel primitive bevel represents the extent of observed pixel cells.
Neither number changes the historical scorer's one-pixel penetration threshold.

Hidden mass cannot be identified from an image when mass is independent of
appearance. The baseline samples the declared prior and retains the same draw
across changed actions. It does not read the reference scene's true physics.
Likewise, a conditional mean of different valid collision paths need not be a
valid collision path. Low average trajectory error alone cannot prove physical
validity or calibrated uncertainty.

## Applied choice and acceptance checks

Implement an observable-hull, sampled-world rigid-body generator as a clearly
labelled **physics-assisted baseline**. Keep learned models and simulator
reference panels identifiable. This is the lowest-cost evidence-backed path to
physically constrained motion in this restricted domain, not a claim of general
state-of-the-art performance or successful neural physics learning.

Before integration, require analytical force/impulse tests, contact momentum
balance, post-pulse energy dissipation, exact rest, numerical convergence,
all-four-seed reproducibility and exact frame-zero preservation. Then compare
all original validation conditions with the same RGB tracker and native hull
penetration measurement. Freeze code and checkpoint identities before opening
a separately registered fresh-scene evaluation. Retain missing tracks, action
failures and all samples. Do not select attractive videos or relax old gates.

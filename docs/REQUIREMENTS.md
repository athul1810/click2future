# Requirements and open questions

## Confirmed from the project brief

- Project: Click2Future, for the NTU Generative AI course.
- Goal: train an actual generative model for action-conditioned tabletop futures.
- Inputs: initial image, selected-object mask, spatial action map, force direction/magnitude.
- Proposed contribution: object-centric action conditioning and trajectory/collision supervision.
- Five planned comparisons: conditional VAE, first-frame diffusion, numeric-action diffusion, spatial conditioning, full supervised model.
- Interactive output: four sampled futures with simulator comparison where metadata is available.
- Scope starts at 64px; 128px is conditional on feasibility.
- Deadline: 14 November 2026, based on the supplied correspondence.
- All task owners are initially unassigned.
- Model training will use the course-provided TC2 GPU Cluster, as confirmed by the team.

## Instructor guidance

The instructor recommends Click2Future with modest model and experimental scope. The course announcement provides TC2 GPU access for MSc coursework through an application form sent to `ccdsgpu-tc@ntu.edu.sg`. T02 covers that process; record the assigned hardware and quota after receiving access details.

## Working design choices to confirm in T01/T04

| Choice | Proposed starting point | Why |
|---|---|---|
| World | Top-down bounded tabletop with 2-4 rigid objects | Reliable labels and manageable appearance |
| Clip | 16 frames at 8 fps, frame 0 before action | Short sequence for initial profiling |
| Action | Constant force applied at selected object's centre for a fixed duration | Unambiguous direction/magnitude; no click-position torque |
| Hidden properties | Mass and friction vary independently of visible appearance | Creates controlled conditional uncertainty |
| Model | Small frame autoencoder + latent video denoiser | Profile feasibility before scaling |
| Demo | Simulator presets plus synthetic uploads | Keeps evaluation within the training domain |

These are proposed engineering defaults, not course requirements. Record the final values and units in the data contract and configs.

## Course-site checklist

- [ ] Confirm course code/title; do not inherit AI6101 from the reference repository.
- [ ] Confirm deadline time/timezone and submission platform.
- [ ] Confirm report page limit, presentation format, and required deliverables.
- [ ] Confirm rules for pretrained models, external datasets, and AI-assisted work.
- [ ] Confirm team-member details and contribution reporting.
- [ ] Record TC2 application/access status, assigned GPU model, memory, quota, storage, job limits, and software environment.

## Completion criteria

- A seeded simulator exports reproducible clips, masks, trajectories, and collision labels.
- At least one compact diffusion model actually trains and produces action-conditioned video.
- The planned baseline ladder is evaluated under documented, comparable settings; any omissions are disclosed and justified.
- Spatial conditioning and auxiliary supervision are tested as separate changes.
- Identical visible inputs with hidden physical variation support a controlled uncertainty experiment.
- Evaluation includes separate shape, texture, and force generalization splits.
- The demo shows four samples and a correctly labelled simulator comparison.
- The report links claims to measured results, includes failure cases, and states limitations.
- A teammate can reproduce a smoke run and load the final checkpoints from the documented environment.

## Scope changes

Record changes in [DECISIONS.md](DECISIONS.md) with the reason, evidence, and effect on experiments. If compute is constrained, reduce resolution, clip length, model size, and dataset size before expanding features. Do not silently remove an ablation or relabel a simulator output as a generated sample.

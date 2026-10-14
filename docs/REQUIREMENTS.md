# Course requirements and project scope

Source: `course project technical review.pdf`. Page numbers refer to the 11-page course brief.

## Confirmed course requirements

| Requirement | Source | Click2Future plan |
|---|---|---|
| AI6132 Generative AI For Visual Synthesis, Project 1: Technical Review | Pages 1-2 | Review and compare action-conditioned video generation methods |
| Up to four members; use full names as on student cards | Page 2 | Group 7 roster recorded in [README.md](../README.md) |
| First person listed in the group is the coordinator | Page 2 | PRAMOD SANJAY, first in the supplied roster |
| Identify a visual generation task | Page 6 | Generate tabletop futures from an initial image and an applied force |
| At least one evaluation dataset; subsets and self-built datasets are allowed | Page 7 | Procedural tabletop dataset with independent scene splits |
| Compare at least two methods; existing and proposed methods are allowed | Page 7 | CVAE and video diffusion, with E1-E4 ablations |
| Analyze hyperparameters and key components | Page 8 | Small validation-only parameter study plus conditioning/supervision ablations |
| Compare methods and discuss strengths, limitations, and future extensions | Page 8 | Quantitative tables, failure cases, and discussion |
| Visualize and discuss results; do not copy results from other materials | Page 8 | Use our measured runs and generated examples |
| Technical report of at least four pages | Page 8 | Introduction, methods, experiments and analysis, conclusion |
| Submit report, contribution PDF, and source-code ZIP | Pages 4-5 | Follow the exact filenames in [SUBMISSION.md](SUBMISSION.md) |
| No datasets, model files, external links, or cloud-hosted submissions | Pages 4-5 | Package source code only and put results directly in the report |
| One submission per group; replacements are allowed without a stated limit | Page 5 | Coordinate one group submission through NTULearn |

The PDF does not require a novel method, five variants, 128px generation, slides, or a separate demo video. Our custom conditioning module and demo are project choices.

## Deadline

**Working deadline: 14 November 2026.** The course brief lists 16 November; confirm the exact cutoff on NTULearn. See [SUBMISSION.md](SUBMISSION.md).

## Rubric and teamwork

Page 9 asks for a clear task/method explanation, good organization, thorough comparison and analysis, insightful discussion, useful visualizations, challenging methods or applications, and discussion or implementation of improvements. It gives no numerical weights.

Everyone must contribute and communicate with the group. Page 10 includes a policy to report a member's refusal to contribute to the instructors at least six weeks before submission. Keep contribution records throughout the project and refer to that policy if a team issue arises.

## Click2Future scope

- Train a compact generative model for action-conditioned tabletop futures.
- Inputs: initial image, selected-object mask, spatial action map, force direction and magnitude.
- Compare a conditional VAE with video diffusion, then test spatial conditioning and trajectory/collision supervision separately.
- Use hidden mass and friction to study uncertainty across four sampled futures.
- Start at 64px; consider 128px only after profiling.
- Use the course-provided TC2 GPU Cluster for training.

## Working design choices

| Choice | Proposed starting point | Reason |
|---|---|---|
| World | Top-down bounded tabletop with 2-4 rigid objects | Exact labels and manageable appearance |
| Clip | 16 frames at 8 fps, frame 0 before action | Short sequence for profiling |
| Action | Constant force at the selected object's centre for a fixed duration | Clear direction and magnitude; no click-position torque |
| Hidden properties | Mass and friction independent of visible appearance | Controlled conditional uncertainty |
| Model | Small frame autoencoder and latent video denoiser | Measure training cost before scaling |
| Demo | Simulator presets and synthetic uploads | Keep the interface within the training domain |

Finalize values and units in T04. These are engineering choices, not requirements from the PDF.

## Remaining details to record

- [x] Group 7, full member names, and coordinator recorded from the supplied roster.
- [ ] Deadline discrepancy, exact cutoff/timezone, and live NTULearn submission link.
- [ ] Any additional report template, page maximum, file-size limit, or updated instructions on NTULearn; none is specified in the course brief.
- [ ] Rules on pretrained weights and AI-assisted work, if given elsewhere; the PDF's ban concerns submitted datasets and model files.
- [x] TC2 access approval for Athul and account resource/job limits recorded in [COMPUTE.md](COMPUTE.md).
- [ ] TC2 first login, GPU model/VRAM, remaining quota details, and software environment.

## Completion criteria

- Compare at least two methods on at least one dataset using our own experiments.
- Include hyperparameter analysis, component analysis, visualizations, limitations, and future work in a report of at least four pages.
- For the planned Click2Future method, validate simulator labels, action conditioning, and uncertainty rather than claim improvements without evidence.
- Keep code, configs, seeds, and internal run records reproducible; verify the final code ZIP separately from the team's checkpoint-based workflow.
- Submit exactly the three named files, with no datasets, model files, external links, or cloud-hosted substitutes.

Record scope changes in [DECISIONS.md](DECISIONS.md). If compute is tight, reduce resolution, clip length, model size, and optional experiments while retaining the course requirements. Clearly report changes to the five-variant plan.

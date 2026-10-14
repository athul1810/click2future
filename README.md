# Click2Future

### Action-conditioned video diffusion for physical futures

Select an object in a tabletop scene, draw a force arrow, and generate four possible outcomes.

Click2Future is our **AI6132 Generative AI For Visual Synthesis** project at NTU. For Project 1: Technical Review, we will compare a conditional VAE and video diffusion for predicting tabletop motion after a force is applied. A procedural simulator will generate the clips, trajectories, and collision labels. We also plan to test object-centric conditioning and auxiliary physics supervision.

**Status:** the development environment and an untrained video-denoiser prototype are
available. The simulator, dataset pipeline, CVAE, diffusion trainer/sampler, evaluation,
and demo still need to be built or integrated.

**Working deadline:** 14 November 2026. See [submission details](docs/SUBMISSION.md) for the course-brief date discrepancy.

**Task owners:** all unassigned; claim work through [GitHub Issues](https://github.com/athul1810/click2future/issues).

**Compute:** use the course-provided TC2 GPU Cluster. Apply using the course form, then record the GPU model, memory, and usage limits before setting the training budget.

## Start here

| If you want to… | Read… |
|---|---|
| Prepare for the 14 October project meeting | [Meeting brief](docs/MEETING_2026-10-14.md) |
| Assess relevant SOTA and choose an achievable build path | [Research review](docs/SOTA_2026-10-14.md) and [build plan](docs/BUILD_PLAN.md) |
| Establish TC2 access and run the first diagnostic job | [TC2 workflow](cluster/README.md) and [application template](docs/TC2_ACCESS.md) |
| Understand the project and scope | [Requirements](docs/REQUIREMENTS.md) |
| Prepare the three required submission files | [Submission guide](docs/SUBMISSION.md) |
| Draft the report and contribution statement | [Report outline](docs/REPORT_OUTLINE.md) and [contribution template](docs/CONTRIBUTION_TEMPLATE.md) |
| Assign work to the team | [Task board](docs/TASKS.md) and [open issues](https://github.com/athul1810/click2future/issues) |
| See the deadline and task dependencies | [Roadmap](docs/ROADMAP.md) |
| Implement compatible components | [Architecture and data contract](docs/ARCHITECTURE.md) |
| Design experiments and metrics | [Evaluation plan](docs/EVALUATION.md) |
| Find papers for the method and report | [Research references](docs/REFERENCES.md) |
| Plan GPU usage | [Compute and feasibility](docs/COMPUTE.md) |
| Contribute a change | [Contribution guide](CONTRIBUTING.md) |

## Research question

> Can a compact video diffusion model generate diverse futures while obeying an explicitly specified physical action?

We will test an **object-centric action-conditioning module** and an auxiliary head that predicts trajectories and collisions from denoiser features. The experiments will check whether each component improves the generated videos.

Hidden mass and friction can produce different outcomes from the same image and action. We will compare generated samples with several simulator rollouts to check whether the variation reflects those physical differences.

## The intended demo

1. Load a simulator scene or upload a supported synthetic tabletop image.
2. Click an object and drag an arrow to specify force direction and magnitude.
3. Generate four futures using different sampling seeds.
4. Compare them side by side with a simulator rollout and trajectory overlays.
5. Adjust the force or simulator friction and repeat.

For the core hidden-physics model, friction is **not an input**. Changing the friction slider changes the simulator comparison; it must not be presented as controlling model samples. A friction-conditioned model is a stretch goal. Simulator ground truth requires scene metadata; image-only uploads cannot provide exact ground truth.

## Scope

| Planned project work | Stretch goal |
|---|---|
| 64×64 top-down scenes, initially 16 frames | 128×128 after the compute gate passes |
| 2-4 rigid objects, randomized appearance and hidden physics | More shapes and harder collisions |
| Compact video diffusion and conditional VAE baseline; pixel pilot proposed | Reconstruction-gated latent architecture |
| Five planned experimental variants | Additional supervision ablations |
| Four sampled futures and a simulator comparison | Friction-conditioned sampling |
| Reproducible experiments, technical report, contribution PDF, source code | FVD if feasible and statistically useful |

The course requires at least two methods, one evaluation dataset, and a report of at least four pages. Five variants and the interactive demo are our project plan, not additional course submission requirements. Checkpoints and any demo recordings stay in the team's working files.

No real robot, photorealistic 512px video, or general-purpose physics. The upload workflow targets the synthetic domain; real photographs are outside the course scope.

## System overview

```mermaid
flowchart LR
    S[Procedural simulator] --> D[Clips, masks, trajectories, collisions]
    D --> V[Compact pixel video denoiser - proposed pilot]
    I[First frame + clicked mask + action] --> C[Object-centric conditioning]
    C --> V
    V --> H[Trajectory and collision heads]
    V --> O[Planned sampler - four generated futures]
    O --> E[Metrics and interactive comparison]
    S --> E
```

## Planned experiments

| ID | Variant | Purpose |
|---|---|---|
| E0 | Conditional VAE with first frame, selected mask, and numeric action | Low-cost generative baseline |
| E1 | First-frame-only video diffusion | Measure what happens without action information |
| E2 | Diffusion with first frame, selected mask, and numeric action | Isolate spatial action encoding from target identification |
| E3 | E2 + spatial action conditioning | Test the proposed conditioning module |
| E4 | E3 + trajectory/collision supervision | Test the effect of auxiliary physics supervision |

Compare ADE/FDE, action-direction error, collision accuracy and F1, SSIM/LPIPS, motion-aware temporal consistency, and trajectory diversity. Evaluate held-out shapes, textures, and force magnitudes separately. [Evaluation details](docs/EVALUATION.md) define fairness controls and the limits of each metric.

## Four workstreams

Divide these areas across the team. Each issue should have one lead and a reviewer.

| Workstream | Main responsibility | Owner |
|---|---|---|
| A: Simulator and data | Physics, rendering, labels, splits, ground-truth replay | Unassigned |
| B: Generative models | Autoencoder, CVAE, diffusion baselines, action module | Unassigned |
| C: Evaluation and experiments | Metrics, supervision integration, ablations, analysis | Unassigned |
| D: Integration and delivery | Demo, reproducibility, compute coordination, report assembly | Unassigned |

Each person writes up the methods, results, and limitations of their work. See the [task board](docs/TASKS.md) for dependencies and acceptance criteria.

## Submission

One submission per group through NTULearn, containing exactly:

- `group_XX_report.pdf`: main report, at least four pages.
- `group_XX_contribution.pdf`: each member's contribution.
- `group_XX_code.zip`: source code, with no datasets, model files, or external links.

Replace `XX` with the confirmed group ID. No cloud-hosted substitutes or external links in the submitted material. Follow the [submission guide](docs/SUBMISSION.md); do not ZIP the working repository wholesale.

## Repository layout

```text
.github/             Issue and pull-request templates
docs/                Scope, task board, roadmap, interfaces, evaluation, compute
src/click2future/     Diagnostics, untrained denoiser, and synthetic profiler
configs/             Development/pilot configs; remaining experiment configs planned
cluster/             Allocation-guarded TC2 inventory/bootstrap/probe jobs
scripts/             Local TC2 network/access preflight
experiments/         Run manifest template and experiment records
results/             Small result tables and report figures
app/                 Planned interactive demo
tests/               Environment and prototype checks; simulator/sampler checks planned
```

See [development setup](docs/SETUP.md) for the locked Python environment, CPU/CUDA checks,
and editor instructions. The package includes diagnostics, an untrained denoiser prototype,
and a synthetic architecture profiler. Simulator, dataset training, sampling, evaluation,
and demo commands are not implemented yet. The [build plan](docs/BUILD_PLAN.md) distinguishes
the tested prototype from planned functionality and records the proposed pixel/latent choice.

## First team meeting

- Assign T01-T03 and reviewers; start T04 after the scope decision. Check the course requirements and complete the TC2 application/access setup.
- Agree on the action semantics and dataset contract before generating the main dataset.
- Claim the remaining issues by workstream and check their dependencies.
- Use the 64px pilot to decide the dataset and training budget; keep 128px optional.

## References

See [Research references](docs/REFERENCES.md) for the baselines, diffusion models, and evaluation metrics.

## Results

No experiments have been run. Add measured results with run IDs, configs, seeds, and checkpoint references to [results/](results/).

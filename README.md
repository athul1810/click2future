# Click2Future

### Action-conditioned video diffusion for physical futures

Select an object in a tabletop scene, draw a force arrow, and generate four possible outcomes.

Click2Future is our **AI6132 Generative AI For Visual Synthesis** project at NTU. For Project 1: Technical Review, we will compare a conditional VAE and video diffusion for predicting tabletop motion after a force is applied. A procedural simulator will generate the clips, trajectories, and collision labels. We also plan to test object-centric conditioning and auxiliary physics supervision.

**Status:** seeded simulation, grouped data generation/replay, the development environment,
and a trainable pixel denoiser are available. Evaluation includes trajectory/collision
metrics, a pixel tracker, zero/reversed-force controls, SSIM/LPIPS, temporal checks,
and ensemble diversity/coverage. Contact extraction remains an approximate proxy
with measured errors.
A dataset trainer, complete checkpoint/resume and seeded DDIM sampler are implemented,
including auxiliary-loss integration. The first 2,000-step learning pilot failed
generated-video checks. The longer epsilon and velocity probes also failed the
quality gate. CVAE training and prior-sampling code are ready for TC2 checks;
trained CVAE results, held-out learned-video controls and the demo remain open.
See [pilot evidence](docs/results/diffusion_pilot_2026-10-16.json)
and [evaluation results](docs/METRICS.md). See [model training](docs/MODEL_TRAINING.md)
for the CVAE and E4 workflow.

**Working submission deadline:** 14 November 2026. See [submission details](docs/SUBMISSION.md) for the course-brief date discrepancy.

**Workstream owners:** A: Sanjay; B: Shrijeet; C: Athul and Ananya; CVAE: Ananya proposed, awaiting acceptance; demo integration: Shrijeet. See the [task board](docs/TASKS.md).

**Compute:** TC2 access is approved for Athul and Shrijeet. A40 setup and real
training jobs passed, including exact next-update continuation across allocations.
The four-clip diagnostic measured 0.04883 s/step and 1.744 s for four 32-step
samples. These timings do not establish video quality or full-dataset cost.
Each account has a one-GPU, six-hour-job limit. See [compute details](docs/COMPUTE.md).

## Group 7

| Member | NTU username |
|---|---|
| PRAMOD SANJAY | sanjay009 |
| KAPOOR ANANYA | kapo0001 |
| MAITI SHRIJEET | shrijeet001 |
| BOBAN ATHUL KRISHNA | athulkri001 |

Coordinator: **PRAMOD SANJAY**, following the course rule that the first person listed in the group is the coordinator.

## Start here

| If you want to… | Read… |
|---|---|
| Execute the plan for the 14 November PPT presentation | [16 October execution plan](docs/EXECUTION_PLAN_2026-10-16.md) and [demo specification](app/DEMO_SPEC.md) |
| Prepare for the 14 October project meeting | [Meeting brief](docs/MEETING_2026-10-14.md) |
| Assess relevant SOTA and choose an achievable build path | [Research review](docs/SOTA_2026-10-14.md) and [build plan](docs/BUILD_PLAN.md) |
| Verify TC2 login and run the first diagnostic job | [TC2 workflow](cluster/README.md) and [access status](docs/TC2_ACCESS.md) |
| Understand the project and scope | [Requirements](docs/REQUIREMENTS.md) |
| Prepare the three required submission files | [Submission guide](docs/SUBMISSION.md) |
| Draft the report and contribution statement | [Report outline](docs/REPORT_OUTLINE.md) and [contribution template](docs/CONTRIBUTION_TEMPLATE.md) |
| Assign work to the team | [Task board](docs/TASKS.md) and [open issues](https://github.com/athul1810/click2future/issues) |
| See the deadline and task dependencies | [Roadmap](docs/ROADMAP.md) |
| Implement compatible components | [Architecture and data contract](docs/ARCHITECTURE.md) |
| Design experiments and run implemented metrics | [Evaluation plan](docs/EVALUATION.md) and [metric usage](docs/METRICS.md) |
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
| 64×64 top-down scenes, 32-frame simulator pilot | 128×128 after the compute gate passes |
| 2-4 rigid objects, randomized appearance and hidden physics | More shapes and harder collisions |
| Compact video diffusion and conditional VAE baseline; pixel path adopted | Reconstruction-gated latent architecture |
| E0/E2/E3 core; E1/E4 gated ablations | Additional supervision ablations |
| Four sampled futures and a simulator comparison | Friction-conditioned sampling |
| Reproducible experiments, technical report, contribution PDF, source code | FVD if feasible and statistically useful |

The course requires at least two methods, one evaluation dataset, and a report of at least four pages. E1/E4 and advanced demo features are gated additions to the core comparison. Checkpoints and any demo recordings stay in the team's working files.

No real robot, photorealistic 512px video, or general-purpose physics. The upload workflow targets the synthetic domain; real photographs are outside the course scope.

## System overview

```mermaid
flowchart LR
    S[Procedural simulator] --> D[Clips, masks, trajectories, collisions]
    D --> V[Compact pixel video denoiser]
    I[First frame + clicked mask + action] --> C[Object-centric conditioning]
    C --> V
    V --> H[Trajectory and collision heads]
    V --> O[DDIM sampler - four generated futures]
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

C is shared by Athul and Ananya. Agree on the implementation split within each shared task and choose reviewers separately.

| Workstream | Main responsibility | Owner |
|---|---|---|
| A: Simulator and data | Physics, rendering, labels, splits, ground-truth replay | PRAMOD SANJAY |
| B: Diffusion | Pixel diffusion training/sampling, numeric and spatial action conditioning | MAITI SHRIJEET |
| C: Evaluation and physics | Tracking, trajectory/collision metrics, uncertainty, experiments, auxiliary supervision with B | BOBAN ATHUL KRISHNA and KAPOOR ANANYA |
| D: CVAE | Prior/posterior baseline and fair sampling/cost comparison | Ananya proposed; acceptance pending |
| Demo integration | Shared sampler, four-future UI, simulator comparison and offline fallback | MAITI SHRIJEET |

Confirm Ananya's CVAE acceptance; see [the handoff](docs/CVAE_HANDOFF.md). Report assembly and submission tasks also remain unassigned. Each person writes up the methods, results, and limitations of their work. See the [task board](docs/TASKS.md) for dependencies and acceptance criteria.

## Submission

One submission per group through NTULearn, containing exactly:

- `group_07_report.pdf`: main report, at least four pages.
- `group_07_contribution.pdf`: each member's contribution.
- `group_07_code.zip`: source code, with no datasets, model files, or external links.

Use `07` for Group 7 in all three filenames. No cloud-hosted substitutes or external links in the submitted material. Follow the [submission guide](docs/SUBMISSION.md); do not ZIP the working repository wholesale.

## Repository layout

```text
.github/             Issue and pull-request templates
docs/                Scope, task board, roadmap, interfaces, evaluation, compute
src/click2future/     Simulation, data pipeline, denoiser prototype, evaluation
configs/             Simulator, development, profiler, and pixel evaluation configs
cluster/             Allocation-guarded TC2 inventory/bootstrap/probe jobs
scripts/             Local TC2 network/access preflight
experiments/         Run manifest template and experiment records
results/             Small result tables and report figures
app/                 Planned interactive demo
tests/               Simulator, data, prototype, and evaluation checks
```

See [development setup](docs/SETUP.md) for the locked environment and
[metric usage](docs/METRICS.md) for RGB evaluation. The [training guide](docs/TRAINING.md) covers the implemented trainer and sampler.
The demo UI remains to be built. The [build plan](docs/BUILD_PLAN.md) records the adopted pixel path.

## Next team work

- A: finish appearance/OOD dataset coverage and measure generation/storage costs.
- B: complete the TC2 learning/resume pilot, inspect generated motion, then pass held-out action controls.
- C: improve contact timing, add the remaining metrics, then integrate auxiliary losses with B.
- Confirm the CVAE lead; Shrijeet owns demo integration.

## References

See [Research references](docs/REFERENCES.md) for the baselines, diffusion models, and evaluation metrics.

## Results

Environment and simulator/metric controls passed. The first learned-video pilot
passed full-state resume but failed generated-video quality. No method ranking
is established. Record run IDs, configs, seeds, failures and checkpoint hashes.

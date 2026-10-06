# Task board

All tasks start **unassigned**. Assign leads and reviewers in [GitHub Issues](https://github.com/athul1810/click2future/issues); that is the live source of ownership and progress. This file records the initial scope and dependencies.

Start with **T01, T02, and T04**. T03 can prepare the CPU environment while GPU details are pending. Demo shell work and the report outline can start early against the agreed contract; their final acceptance requires integrated evidence.

| ID | Task | Workstream | Milestone | Depends on | Initial owner |
|---|---|---|---|---|---|
| T01 | [Confirm course requirements and freeze scope](https://github.com/athul1810/click2future/issues/1) | planning | M1 | — | Unassigned |
| T02 | [Confirm TC2 access and record the compute budget](https://github.com/athul1810/click2future/issues/2) | infrastructure | M1 | — | Unassigned |
| T03 | [Create a reproducible development environment](https://github.com/athul1810/click2future/issues/3) | infrastructure | M1 | T02 | Unassigned |
| T04 | [Freeze simulator, action, and dataset interfaces](https://github.com/athul1810/click2future/issues/4) | data | M1 | T01 | Unassigned |
| T05 | [Implement the seeded 64px tabletop simulator](https://github.com/athul1810/click2future/issues/5) | data | M2 | T03, T04 | Unassigned |
| T06 | [Build versioned datasets and leakage-safe splits](https://github.com/athul1810/click2future/issues/6) | data | M2 | T05 | Unassigned |
| T07 | [Implement and validate the evaluation pipeline](https://github.com/athul1810/click2future/issues/7) | evaluation | M2 | T04, T05 | Unassigned |
| T08 | [Train and validate the shared frame autoencoder](https://github.com/athul1810/click2future/issues/8) | model | M3 | T03, T06 | Unassigned |
| T09 | [Train the conditional VAE baseline E0](https://github.com/athul1810/click2future/issues/9) | model | M3 | T03, T06 | Unassigned |
| T10 | [Implement first-frame and numeric-action diffusion E1/E2](https://github.com/athul1810/click2future/issues/10) | model | M3 | T08 | Unassigned |
| T11 | [Implement object-centric spatial conditioning E3](https://github.com/athul1810/click2future/issues/11) | model | M4 | T10, T04 | Unassigned |
| T12 | [Add trajectory and collision supervision E4](https://github.com/athul1810/click2future/issues/12) | model | M4 | T11, T07 | Unassigned |
| T13 | [Freeze experiment settings and decide the 128px gate](https://github.com/athul1810/click2future/issues/13) | evaluation | M4 | T07, T09, T10, T11, T12 | Unassigned |
| T14 | [Run and analyze the core E0–E4 comparisons](https://github.com/athul1810/click2future/issues/14) | evaluation | M5 | T13 | Unassigned |
| T15 | [Evaluate held-out shapes, textures, and forces](https://github.com/athul1810/click2future/issues/15) | evaluation | M5 | T13 | Unassigned |
| T16 | [Evaluate hidden-physics uncertainty and action controls](https://github.com/athul1810/click2future/issues/16) | evaluation | M5 | T13 | Unassigned |
| T17 | [Build the interactive four-future comparison demo](https://github.com/athul1810/click2future/issues/17) | demo | M5 | T04, T05, T10 | Unassigned |
| T18 | [Prepare the report draft and evidence figures](https://github.com/athul1810/click2future/issues/18) | delivery | M5 | T01, T13 | Unassigned |
| T19 | [Verify reproducibility and freeze the submission artifacts](https://github.com/athul1810/click2future/issues/19) | delivery | M6 | T14, T15, T16, T17 | Unassigned |
| T20 | [Finalize report, slides, and submission package](https://github.com/athul1810/click2future/issues/20) | delivery | M6 | T18, T19 | Unassigned |

## Task details

<a id="t01"></a>

### T01 — Confirm course requirements and freeze scope

Record the official course constraints and agree on a modest, testable project.

**Milestone:** M1 · **Dependencies:** None · **Owner:** Unassigned

- [ ] Complete the course-site checklist in docs/REQUIREMENTS.md, including the exact deadline cutoff.
- [ ] Confirm the research question, core deliverables, and optional features.
- [ ] Record decisions and unresolved questions in docs/DECISIONS.md.

<a id="t02"></a>

### T02 — Confirm TC2 access and record the compute budget

Establish the hardware and scheduling limits before promising training runs.

**Milestone:** M1 · **Dependencies:** None · **Owner:** Unassigned

- [ ] Check the coursework GPU application and record allocation status.
- [ ] Record GPU/VRAM, job duration, storage, and software constraints in docs/COMPUTE.md.
- [ ] Define a profiling plan and fallback if allocation remains pending.

<a id="t03"></a>

### T03 — Create a reproducible development environment

Make setup reproducible on the development machine and allocated training hardware.

**Milestone:** M1 · **Dependencies:** T02 · **Owner:** Unassigned

- [ ] Add a dependency manifest and tested setup instructions with exact versions.
- [ ] Add a small environment/device check and document CPU versus GPU workflows.
- [ ] Select artifact locations and a configuration format; keep credentials out of Git.
- [ ] Document working commands as modules are implemented; mark unavailable commands clearly.

<a id="t04"></a>

### T04 — Freeze simulator, action, and dataset interfaces

Let simulator, model, and evaluation work proceed against one agreed contract.

**Milestone:** M1 · **Dependencies:** T01 · **Owner:** Unassigned

- [ ] Finalize units, axis conventions, force duration, friction law, frame timing, and collision event semantics.
- [ ] Finalize tensor shapes, object identity/padding, metadata, and conditioning inputs in docs/ARCHITECTURE.md.
- [ ] Define scene_group_id and condition_id, including identical-first-frame uncertainty groups.
- [ ] Provide a small fixture or schema example that downstream components can consume.

<a id="t05"></a>

### T05 — Implement the seeded 64px tabletop simulator

Generate repeatable tabletop clips with usable physical labels.

**Milestone:** M2 · **Dependencies:** T03, T04 · **Owner:** Unassigned

- [ ] Render 2–4 objects with randomized supported shapes/textures, mass, and friction.
- [ ] Apply the agreed centre force and export positions, masks, object collisions, and wall contacts.
- [ ] Verify deterministic replay, zero-force behavior, force reversal, and a known collision case.
- [ ] Provide sample clips and a documented generation/replay command.

<a id="t06"></a>

### T06 — Build versioned datasets and leakage-safe splits

Produce a traceable dataset with independent visible-scene groups and controlled OOD tests.

**Milestone:** M2 · **Dependencies:** T05 · **Owner:** Unassigned

- [ ] Export a smoke dataset, measure generation cost/storage, and propose the main dataset size.
- [ ] Split by scene_group_id; keep all actions and physics variants of a visible scene together.
- [ ] Create ID, shape-OOD, texture-OOD, force-OOD, and repeated-hidden-physics test manifests.
- [ ] Check identical initial inputs within condition_id and independence of hidden physics from appearance.
- [ ] Publish dataset/schema versions, split hashes, and download or generation instructions.

<a id="t07"></a>

### T07 — Implement and validate the evaluation pipeline

Measure generated video behavior consistently across models.

**Milestone:** M2 · **Dependencies:** T04, T05 · **Owner:** Unassigned

- [ ] Implement ADE/FDE, collision accuracy/precision/recall/F1, and image/temporal/diversity metrics.
- [ ] Validate the object tracker and contact estimator on held-out simulator clips; record extraction failures.
- [ ] Test metrics on exact, shifted, static, and corrupted controls with expected outcomes.
- [ ] Separate video-derived metrics from auxiliary-head metrics and export machine-readable records.

<a id="t08"></a>

### T08 — Train and validate the shared frame autoencoder

Obtain a compact latent representation that preserves object geometry.

**Milestone:** M3 · **Dependencies:** T03, T06 · **Owner:** Unassigned

- [ ] Implement or select a permitted autoencoder and record architecture/weight provenance.
- [ ] Inspect reconstruction quality on ID and held-out appearance cases.
- [ ] Record memory, throughput, latent shapes, and a checkpoint with checksum.
- [ ] Freeze and share the selected autoencoder across diffusion comparisons.

<a id="t09"></a>

### T09 — Train the conditional VAE baseline E0

Establish a lower-cost generative baseline with the agreed action inputs.

**Milestone:** M3 · **Dependencies:** T03, T06 · **Owner:** Unassigned

- [ ] Condition on the first frame, selected-object mask, and numeric force.
- [ ] Pass a tiny-batch overfit check and produce four seeded samples per condition.
- [ ] Log validation results, model size, training cost, and checkpoint selection.
- [ ] Add a run manifest and qualitative successes/failures.

<a id="t10"></a>

### T10 — Implement first-frame and numeric-action diffusion E1/E2

Create the common diffusion backbone and the main numeric-action baseline.

**Milestone:** M3 · **Dependencies:** T08 · **Owner:** Unassigned

- [ ] Implement latent video training and sampling with a shared backbone/configuration.
- [ ] Support first-frame-only E1 and selected-mask-plus-numeric-force E2.
- [ ] Pass forward/backward, tiny-batch overfit, and checkpoint resume checks.
- [ ] Measure 64px peak VRAM, seconds/step, and four-sample inference latency.
- [ ] Record baseline samples and proposed run budgets.

<a id="t11"></a>

### T11 — Implement object-centric spatial conditioning E3

Test whether localizing the action improves physical action adherence.

**Milestone:** M4 · **Dependencies:** T10, T04 · **Owner:** Unassigned

- [ ] Fuse a selected-object feature, numeric force embedding, and spatial action map into the denoiser.
- [ ] Keep target identification available in E2 so E2-to-E3 isolates the spatial module.
- [ ] Check selected-object changes, zero action, and reversed action with fixed sampling seeds.
- [ ] Record added parameter count, ablation switch, and working E3 training/sampling.

<a id="t12"></a>

### T12 — Add trajectory and collision supervision E4

Test whether physical supervision improves decoded future behavior.

**Milestone:** M4 · **Dependencies:** T11, T07 · **Owner:** Unassigned

- [ ] Predict trajectories and object-pair collision probabilities from denoiser features.
- [ ] Define timestep supervision, object-slot matching, valid masks, imbalance handling, and loss weights.
- [ ] Keep oracle trajectories/collisions/physics out of inference conditioning.
- [ ] Validate loss gradients and tiny-batch behavior; demonstrate E4 training and sampling.
- [ ] Report head predictions separately from generated-video tracking metrics.

<a id="t13"></a>

### T13 — Freeze experiment settings and decide the 128px gate

Commit to a comparable experiment suite that fits the remaining budget.

**Milestone:** M4 · **Dependencies:** T07, T09, T10, T11, T12 · **Owner:** Unassigned

- [ ] Freeze dataset versions, model controls, sampler, K, metrics, seeds, and checkpoint criteria.
- [ ] Use measured costs to allocate training/evaluation hours and retry reserve.
- [ ] Record a justified proceed/defer decision for 128px; preserve core 64px comparisons.
- [ ] Record all selected settings and validation-only tuning decisions.

<a id="t14"></a>

### T14 — Run and analyze the core E0–E4 comparisons

Produce the main evidence for conditioning and supervision claims.

**Milestone:** M5 · **Dependencies:** T13 · **Owner:** Unassigned

- [ ] Run the five variants under the frozen protocol and record actual budgets/seed counts.
- [ ] Compare E2-to-E3 and E3-to-E4 with quality, physical behavior, and cost metrics.
- [ ] Report mean and best-of-K trajectory scores separately, plus collision prevalence/F1.
- [ ] Export tables with run IDs, sample counts, group-based uncertainty estimates, and failures.

<a id="t15"></a>

### T15 — Evaluate held-out shapes, textures, and forces

Measure which action-conditioned behaviors generalize beyond training.

**Milestone:** M5 · **Dependencies:** T13 · **Owner:** Unassigned

- [ ] Evaluate separate OOD suites while controlling the other factors.
- [ ] Use frozen checkpoints and metrics without test-set tuning.
- [ ] Compare against ID performance and include tracker failure rates.
- [ ] Save representative successes/failures and state generalization limits.

<a id="t16"></a>

### T16 — Evaluate hidden-physics uncertainty and action controls

Distinguish physically meaningful uncertainty from visual noise.

**Milestone:** M5 · **Dependencies:** T13 · **Owner:** Unassigned

- [ ] Sample K=4 model futures for identical visible conditions and compare with multiple simulator rollouts.
- [ ] Measure trajectory spread, reference coverage, sample plausibility, and collision-frequency variation.
- [ ] Evaluate zero force, reversed force, and selected-object changes.
- [ ] Present quality/adherence alongside diversity and clearly label single-reference metrics.

<a id="t17"></a>

### T17 — Build the interactive four-future comparison demo

Make the model and simulator comparison understandable and repeatable.

**Milestone:** M5 · **Dependencies:** T04, T05, T10 · **Owner:** Unassigned

- [ ] Support preset scenes and synthetic-domain image input with selected-object masks.
- [ ] Map drag arrows to the agreed force convention and generate four seeded futures.
- [ ] Show simulator ground truth only when scene metadata exists.
- [ ] Label the friction slider as simulator-only for hidden-physics checkpoints.
- [ ] Add trajectory overlays, loading/error states, and a documented launch path.

<a id="t18"></a>

### T18 — Prepare the report draft and evidence figures

Assemble the research narrative while experiments finish.

**Milestone:** M5 · **Dependencies:** T01, T13 · **Owner:** Unassigned

- [ ] Draft the question, related work with primary references, method, and experiment protocol.
- [ ] Create table/figure slots tied to E0–E4, OOD, uncertainty, compute, and limitations.
- [ ] Insert measured findings from T14–T16 as available; mark pending results explicitly.
- [ ] Collect method and contribution notes from each workstream.

<a id="t19"></a>

### T19 — Verify reproducibility and freeze the submission artifacts

Ensure another teammate can reproduce the core workflow and use the final checkpoints.

**Milestone:** M6 · **Dependencies:** T14, T15, T16, T17 · **Owner:** Unassigned

- [ ] Run a clean-environment setup, simulator replay, checkpoint load, and sampling smoke test.
- [ ] Verify configs, split hashes, artifact paths/checksums, and actual run commands.
- [ ] Freeze final tables/figures and record limitations or omitted optional experiments.
- [ ] Record a backup demo and check the interface on the presentation machine.

<a id="t20"></a>

### T20 — Finalize report, slides, and submission package

Deliver the project against the confirmed course requirements.

**Milestone:** M6 · **Dependencies:** T18, T19 · **Owner:** Unassigned

- [ ] Finalize conclusions using measured results, include failures and contribution statements.
- [ ] Complete references, formatting, figures, and presentation rehearsal.
- [ ] Check required code/checkpoints/report/slides/video artifacts against the official rubric.
- [ ] Submit by the confirmed 14 November cutoff and record the submission receipt.

The machine-readable initial backlog is [task_backlog.json](task_backlog.json). See [the roadmap](ROADMAP.md) for milestone dates.

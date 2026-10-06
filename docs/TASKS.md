# Task board

All tasks start **unassigned**. Assign leads and reviewers in [GitHub Issues](https://github.com/athul1810/click2future/issues); that is the live source of ownership and progress. This file records the initial scope and dependencies.

Start with **T01, T02, and T03**; T04 follows the scope decision in T01. CPU setup can proceed while TC2 access is being arranged. Demo shell work and the report outline can start early against the agreed contract; their final acceptance requires integrated evidence.

Dependencies below are prerequisites for completing a task. Outlines, interface reviews, and fixture-based work may start earlier. T02/T03 cover the TC2 application, access setup, and GPU validation.


The three required submission files and archive exclusions are in [SUBMISSION.md](SUBMISSION.md). Course requirements are separate from optional project features.


| ID | Task | Workstream | Depends on | Initial owner |
|---|---|---|---|---|
| T01 | [Record group details and finalize project scope](https://github.com/athul1810/click2future/issues/1) | planning | None | Unassigned |
| T02 | [Complete TC2 application and record access details](https://github.com/athul1810/click2future/issues/2) | infrastructure | None | Unassigned |
| T03 | [Create a reproducible development environment](https://github.com/athul1810/click2future/issues/3) | infrastructure | None | Unassigned |
| T04 | [Freeze simulator, action, and dataset interfaces](https://github.com/athul1810/click2future/issues/4) | data | T01 | Unassigned |
| T05 | [Implement the seeded 64px tabletop simulator](https://github.com/athul1810/click2future/issues/5) | data | T03, T04 | Unassigned |
| T06 | [Build versioned datasets and leakage-safe splits](https://github.com/athul1810/click2future/issues/6) | data | T05 | Unassigned |
| T07 | [Implement and validate the evaluation pipeline](https://github.com/athul1810/click2future/issues/7) | evaluation | T04, T05 | Unassigned |
| T08 | [Train and validate the shared frame autoencoder](https://github.com/athul1810/click2future/issues/8) | model | T03, T06 | Unassigned |
| T09 | [Train the conditional VAE baseline E0](https://github.com/athul1810/click2future/issues/9) | model | T03, T06 | Unassigned |
| T10 | [Implement first-frame and numeric-action diffusion E1/E2](https://github.com/athul1810/click2future/issues/10) | model | T08 | Unassigned |
| T11 | [Implement object-centric spatial conditioning E3](https://github.com/athul1810/click2future/issues/11) | model | T10, T04 | Unassigned |
| T12 | [Add trajectory and collision supervision E4](https://github.com/athul1810/click2future/issues/12) | model | T11, T07 | Unassigned |
| T13 | [Freeze experiment settings and decide the 128px gate](https://github.com/athul1810/click2future/issues/13) | evaluation | T07, T09, T10, T11, T12 | Unassigned |
| T14 | [Run and analyze the core E0-E4 comparisons](https://github.com/athul1810/click2future/issues/14) | evaluation | T13 | Unassigned |
| T15 | [Evaluate held-out shapes, textures, and forces](https://github.com/athul1810/click2future/issues/15) | evaluation | T13 | Unassigned |
| T16 | [Evaluate hidden-physics uncertainty and action controls](https://github.com/athul1810/click2future/issues/16) | evaluation | T13 | Unassigned |
| T17 | [Build the interactive four-future comparison demo](https://github.com/athul1810/click2future/issues/17) | demo | T04, T05, T10 | Unassigned |
| T18 | [Draft the technical report and contribution statement](https://github.com/athul1810/click2future/issues/18) | delivery | T01, T13 | Unassigned |
| T19 | [Verify reproducibility and prepare the source-code ZIP](https://github.com/athul1810/click2future/issues/19) | delivery | T14, T15, T16, T17 | Unassigned |
| T20 | [Finalize the two PDFs and submit the three required files](https://github.com/athul1810/click2future/issues/20) | delivery | T18, T19 | Unassigned |

## Task details

<a id="t01"></a>

### T01: Record group details and finalize project scope

Apply the AI6132 technical-review requirements and record the remaining group and submission details.

**Dependencies:** None · **Owner:** Unassigned

- [ ] Record the group ID, full student-card names, and coordinator from the course group form.
- [ ] Confirm the NTULearn cutoff and resolve the difference between the working 14 November date and the course brief's 16 November date.
- [ ] Check NTULearn for additional instructions and use docs/REQUIREMENTS.md for the confirmed minimum requirements.
- [ ] Agree on the Click2Future scope, preserving at least two compared methods and one evaluation dataset; record any change to the five-variant plan.

<a id="t02"></a>

### T02: Complete TC2 application and record access details

Arrange access to the course-provided TC2 GPU Cluster and document its training limits.

**Dependencies:** None · **Owner:** Unassigned

- [ ] Obtain and complete the application form attached to the course announcement; send it to ccdsgpu-tc@ntu.edu.sg following the form instructions.
- [ ] Record application and access status, then verify a small GPU job after support provides access instructions; keep personal details and credentials outside Git.
- [ ] Record the assigned GPU model/VRAM, usage quota, job duration, storage, and software constraints in docs/COMPUTE.md.
- [ ] Define a profiling plan and size the model and experiments for the assigned TC2 resources.

<a id="t03"></a>

### T03: Create a reproducible development environment

Make the CPU development environment reproducible and document training setup on TC2.

**Dependencies:** None · **Owner:** Unassigned

- [ ] Add a dependency manifest and tested setup instructions with exact versions.
- [ ] Add an environment/device check; verify CPU setup independently and validate TC2 setup after access details are recorded in T02.
- [ ] Select artifact locations and a configuration format; keep credentials out of Git.
- [ ] Document working commands as modules are implemented; mark unavailable commands clearly.

<a id="t04"></a>

### T04: Freeze simulator, action, and dataset interfaces

Let simulator, model, and evaluation work proceed against one agreed contract.

**Dependencies:** T01 · **Owner:** Unassigned

- [ ] Finalize units, axis conventions, force duration, friction law, frame timing, and collision event semantics.
- [ ] Finalize tensor shapes, timestamps, scene replay metadata, object identity/padding, and conditioning inputs in docs/ARCHITECTURE.md.
- [ ] Define scene_group_id and condition_id, including identical-first-frame uncertainty groups.
- [ ] Provide a small fixture or schema example that downstream components can consume.

<a id="t05"></a>

### T05: Implement the seeded 64px tabletop simulator

Generate repeatable tabletop clips with usable physical labels.

**Dependencies:** T03, T04 · **Owner:** Unassigned

- [ ] Render 2-4 objects with randomized supported shapes/textures, mass, and friction.
- [ ] Apply the agreed centre force and export positions, masks, object collisions, and wall contacts.
- [ ] Verify deterministic replay, zero-force behavior, force reversal, and a known collision case.
- [ ] Replay a saved scene using its specification, config, assets, physics, and action; provide a documented generation/replay command and tolerance.

<a id="t06"></a>

### T06: Build versioned datasets and leakage-safe splits

Produce a traceable dataset with independent visible-scene groups and controlled OOD tests.

**Dependencies:** T05 · **Owner:** Unassigned

- [ ] Export a smoke dataset, measure generation cost/storage, and propose the main dataset size.
- [ ] Split by scene_group_id; keep all actions and physics variants of a visible scene together.
- [ ] Create train, validation, ID-test, shape-OOD, texture-OOD, force-OOD, and repeated-hidden-physics test manifests with disjoint scene groups.
- [ ] Check identical initial inputs within condition_id and independence of hidden physics from appearance.
- [ ] Record dataset/schema versions and split hashes; provide procedural generation code, configs, and seeds so the submitted code needs no packaged dataset or external data link.

<a id="t07"></a>

### T07: Implement and validate the evaluation pipeline

Measure generated video behavior consistently across models.

**Dependencies:** T04, T05 · **Owner:** Unassigned

- [ ] Implement ADE/FDE, action-direction and zero-force controls, interval collision accuracy, event precision/recall/F1, and image/temporal/diversity metrics.
- [ ] Validate the object tracker and contact estimator on held-out simulator clips; record extraction failures.
- [ ] Test metrics on exact, shifted, static, and corrupted controls with expected outcomes.
- [ ] Separate video-derived metrics from auxiliary-head metrics and export machine-readable records.
- [ ] Define whole-sample best-of-K selection, metric preprocessing, missing-track handling, and collision-aware temporal checks before evaluating models.

<a id="t08"></a>

### T08: Train and validate the shared frame autoencoder

Obtain a compact latent representation that preserves object geometry.

**Dependencies:** T03, T06 · **Owner:** Unassigned

- [ ] Implement or select a permitted autoencoder and record architecture/weight provenance.
- [ ] Fit weights and normalization on training data only; select the checkpoint on validation data and inspect test/OOD reconstruction only after freezing it.
- [ ] Record memory, throughput, latent shapes, and a checkpoint with checksum.
- [ ] Freeze and share the selected autoencoder across diffusion comparisons.

<a id="t09"></a>

### T09: Train the conditional VAE baseline E0

Establish a lower-cost generative baseline with the agreed action inputs.

**Dependencies:** T03, T06 · **Owner:** Unassigned

- [ ] Condition on the first frame, selected-object mask, and numeric force.
- [ ] Define the posterior, prior, reconstruction loss, and KL weight; sample from the prior without future frames at inference.
- [ ] Pass a tiny-batch overfit check and produce four seeded samples per condition.
- [ ] Log validation results, model size, training cost, and checkpoint selection.
- [ ] Add a run manifest and qualitative successes/failures.

<a id="t10"></a>

### T10: Implement first-frame and numeric-action diffusion E1/E2

Create the common diffusion backbone and the main numeric-action baseline.

**Dependencies:** T08 · **Owner:** Unassigned

- [ ] Implement latent video training and sampling with a shared backbone/configuration.
- [ ] Support first-frame-only E1 and selected-mask-plus-numeric-force E2.
- [ ] Pass forward/backward, tiny-batch overfit, and checkpoint resume checks.
- [ ] Measure 64px peak VRAM, seconds/step, and four-sample inference latency.
- [ ] Record baseline samples and proposed run budgets.

<a id="t11"></a>

### T11: Implement object-centric spatial conditioning E3

Test whether localizing the action improves physical action adherence.

**Dependencies:** T10, T04 · **Owner:** Unassigned

- [ ] Fuse a selected-object feature, numeric force embedding, and spatial action map into the denoiser.
- [ ] Keep target identification available in E2 so E2-to-E3 isolates the spatial module.
- [ ] Check selected-object changes, zero/reversed action, and empty or invalid masks with fixed sampling seeds.
- [ ] Record added parameter count, ablation switch, and working E3 training/sampling.

<a id="t12"></a>

### T12: Add trajectory and collision supervision E4

Test whether physical supervision improves decoded future behavior.

**Dependencies:** T11, T07 · **Owner:** Unassigned

- [ ] Predict trajectories and object-pair collision probabilities from denoiser features.
- [ ] Define timestep supervision, object-slot matching, valid masks, imbalance handling, and loss weights.
- [ ] Keep oracle trajectories/collisions/physics out of inference conditioning; any extra object extraction must use first-frame pixels only.
- [ ] Validate loss gradients and tiny-batch behavior; demonstrate E4 training and sampling.
- [ ] Report head predictions separately from generated-video tracking metrics.

<a id="t13"></a>

### T13: Freeze experiment settings and decide the 128px gate

Commit to a comparable experiment suite that fits the remaining budget.

**Dependencies:** T07, T09, T10, T11, T12 · **Owner:** Unassigned

- [ ] Freeze dataset versions, model controls, sampler, K, metrics, seeds, and checkpoint criteria.
- [ ] Use measured costs to allocate training/evaluation hours and retry reserve.
- [ ] Record a justified proceed/defer decision for 128px; preserve core 64px comparisons.
- [ ] Record all selected settings and validation-only tuning decisions.
- [ ] Use independent runs from a common initialization policy for the main diffusion comparisons; report inherited training separately for warm-started runs.
- [ ] Reserve a small validation-only hyperparameter study, such as sampling-step counts, and record the tested values and compute budget.

<a id="t14"></a>

### T14: Run and analyze the core E0-E4 comparisons

Produce the main evidence for conditioning and supervision claims.

**Dependencies:** T13 · **Owner:** Unassigned

- [ ] Run the five variants under the frozen protocol and record actual budgets/seed counts.
- [ ] Compare E2-to-E3 and E3-to-E4 with quality, physical behavior, and cost metrics.
- [ ] Report mean and best-of-K trajectory scores separately, plus collision prevalence/F1.
- [ ] Export tables with run IDs, sample counts, group-based uncertainty estimates, and failures.
- [ ] Run the planned hyperparameter study and analyze its quality/cost tradeoff; include strengths, limitations, and visual examples from our own runs.

<a id="t15"></a>

### T15: Evaluate held-out shapes, textures, and forces

Measure which action-conditioned behaviors generalize beyond training.

**Dependencies:** T13 · **Owner:** Unassigned

- [ ] Evaluate separate OOD suites while controlling the other factors.
- [ ] Use frozen checkpoints and metrics without test-set tuning.
- [ ] Compare against ID performance and include tracker failure rates.
- [ ] Save representative successes/failures and state generalization limits.

<a id="t16"></a>

### T16: Evaluate hidden-physics uncertainty and action controls

Distinguish physically meaningful uncertainty from visual noise.

**Dependencies:** T13 · **Owner:** Unassigned

- [ ] Sample K=4 model futures for identical visible conditions and compare with multiple simulator rollouts.
- [ ] Measure trajectory spread, reference coverage, sample plausibility, and collision-frequency variation.
- [ ] Evaluate zero force, reversed force, and selected-object changes using the fixed action-control protocol and report stationary/tracking failures.
- [ ] Present quality/adherence alongside diversity and clearly label single-reference metrics.

<a id="t17"></a>

### T17: Build the interactive four-future comparison demo

Make the model and simulator comparison understandable and repeatable.

**Dependencies:** T04, T05, T10 · **Owner:** Unassigned

- [ ] Support preset scenes and synthetic-domain image input with selected-object masks.
- [ ] Map drag arrows to the agreed force convention and generate four seeded futures.
- [ ] Show simulator ground truth only when scene metadata exists.
- [ ] Label the friction slider as simulator-only for hidden-physics checkpoints.
- [ ] Add trajectory overlays, loading/error states, and a documented launch path.

<a id="t18"></a>

### T18: Draft the technical report and contribution statement

Prepare the two required PDFs using the course structure and the team's measured results.

**Dependencies:** T01, T13 · **Owner:** Unassigned

- [ ] Draft at least four pages using introduction, methods, experiments and analysis, and conclusion; follow docs/REPORT_OUTLINE.md.
- [ ] Compare at least two methods on at least one dataset and explain the implemented methods and related work without unsupported novelty claims.
- [ ] Include method comparison, hyperparameter analysis, component analysis, visualizations, strengths, limitations, and future extensions using our own measured runs.
- [ ] Draft each member's actual contribution using docs/CONTRIBUTION_TEMPLATE.md; collect reviewed descriptions rather than invent shares.
- [ ] Embed assessed figures and tables directly in the report; use bibliography entries without external links in both submitted PDFs.

<a id="t19"></a>

### T19: Verify reproducibility and prepare the source-code ZIP

Prepare a source-only submission package and check that it can be used without the team's working files.

**Dependencies:** T14, T15, T16, T17 · **Owner:** Unassigned

- [ ] Run internal simulator replay, checkpoint loading, and sampling checks; retain all model weights and datasets outside the submission.
- [ ] Build group_XX_code.zip from an explicit source/config/dependency-file list, with local setup and run instructions.
- [ ] Include procedural data-generation and seed/split logic, model definitions, training, and evaluation source; exclude datasets, model files, external URLs, and cloud-artifact pointers.
- [ ] Extract the ZIP in a clean folder and test imports, documented commands, and locally generated smoke inputs without existing datasets or checkpoints.
- [ ] Inspect the archive for Git history, environments, caches, logs, credentials, and working documents with links; follow docs/SUBMISSION.md.
- [ ] Keep optional demo recordings and checkpoint-based demo checks internal; document how to regenerate data and train models from the submitted source.

<a id="t20"></a>

### T20: Finalize the two PDFs and submit the three required files

Complete one group submission through NTULearn using the exact required filenames and content restrictions.

**Dependencies:** T18, T19 · **Owner:** Unassigned

- [ ] Export group_XX_report.pdf with at least four pages and the required sections/analysis; inspect every page for readable figures, tables, and references.
- [ ] Export group_XX_contribution.pdf with every member's full name and reviewed contribution description.
- [ ] Use the confirmed group ID in all three filenames and verify group_XX_code.zip contains source code only, with no datasets, model files, or external links.
- [ ] Check both PDFs for external URLs and hyperlinks; include all assessed material directly rather than use cloud-hosted substitutes.
- [ ] Coordinate one group submission of the three files through NTULearn > Assignments > Project 1, using the retained 14 November working deadline until the discrepancy is resolved.
- [ ] Keep the submission receipt privately; if replacing a submission, coordinate the update so the latest three files are consistent. Do not upload slides, checkpoints, or a demo video as extra required files.

The machine-readable initial backlog is [task_backlog.json](task_backlog.json). See [the roadmap](ROADMAP.md) for the working deadline and task dependencies. Intermediate task dates have not been set.

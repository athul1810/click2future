# Task board

A is owned by PRAMOD SANJAY, B by MAITI SHRIJEET, and C jointly by BOBAN ATHUL KRISHNA and KAPOOR ANANYA. D and the remaining setup/delivery tasks are unassigned. Reviewers are still to be chosen. Keep ownership and progress consistent with [GitHub Issues](https://github.com/athul1810/click2future/issues).

The simulator contract is complete through PR #24. Simulator/data work and the local
development environment are implemented in part. C has metric controls, comparison-table
export, and auxiliary losses ready. T07's implementation checks are complete and ready
for review; T12 needs B's dataset trainer and sampling integration. CVAE training/sampling
code is ready for TC2 checks, while D remains unassigned. Dependencies below govern completion, while
fixture-based work can proceed earlier.


The three required submission files and archive exclusions are in [SUBMISSION.md](SUBMISSION.md). Course requirements are separate from optional project features.


| ID | Task | Workstream | Depends on | Owner |
|---|---|---|---|---|
| T01 | [Record group details and finalize project scope](https://github.com/athul1810/click2future/issues/1) | planning | None | Unassigned |
| T02 | [Verify TC2 login and GPU training setup](https://github.com/athul1810/click2future/issues/2) | infrastructure | None | Unassigned |
| T03 | [Create a reproducible development environment](https://github.com/athul1810/click2future/issues/3) | infrastructure | None | Unassigned |
| T04 | [Freeze simulator, action, and dataset interfaces](https://github.com/athul1810/click2future/issues/4) | data | T01 | PRAMOD SANJAY |
| T05 | [Implement the seeded 64px tabletop simulator](https://github.com/athul1810/click2future/issues/5) | data | T03, T04 | PRAMOD SANJAY |
| T06 | [Build versioned datasets and leakage-safe splits](https://github.com/athul1810/click2future/issues/6) | data | T05 | PRAMOD SANJAY |
| T07 | [Implement and validate the evaluation pipeline](https://github.com/athul1810/click2future/issues/7) | evaluation | T04, T05 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA |
| T08 | [Train and validate the shared frame autoencoder](https://github.com/athul1810/click2future/issues/8) | model | T03, T06 | MAITI SHRIJEET |
| T09 | [Train the conditional VAE baseline E0](https://github.com/athul1810/click2future/issues/9) | model | T03, T06 | Unassigned |
| T10 | [Implement first-frame and numeric-action diffusion E1/E2](https://github.com/athul1810/click2future/issues/10) | model | T08 | MAITI SHRIJEET |
| T11 | [Implement object-centric spatial conditioning E3](https://github.com/athul1810/click2future/issues/11) | model | T10, T04 | MAITI SHRIJEET |
| T12 | [Add trajectory and collision supervision E4](https://github.com/athul1810/click2future/issues/12) | model | T11, T07 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA |
| T13 | [Freeze experiment settings and decide the 128px gate](https://github.com/athul1810/click2future/issues/13) | evaluation | T07, T09, T10, T11, T12 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA |
| T14 | [Run and analyze the core E0-E4 comparisons](https://github.com/athul1810/click2future/issues/14) | evaluation | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA |
| T15 | [Evaluate held-out shapes, textures, and forces](https://github.com/athul1810/click2future/issues/15) | evaluation | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA |
| T16 | [Evaluate hidden-physics uncertainty and action controls](https://github.com/athul1810/click2future/issues/16) | evaluation | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA |
| T17 | [Build the interactive four-future comparison demo](https://github.com/athul1810/click2future/issues/17) | demo | T04, T05, T10 | Unassigned |
| T18 | [Draft the technical report and contribution statement](https://github.com/athul1810/click2future/issues/18) | delivery | T01, T13 | Unassigned |
| T19 | [Verify reproducibility and prepare the source-code ZIP](https://github.com/athul1810/click2future/issues/19) | delivery | T14, T15, T16, T17 | Unassigned |
| T20 | [Finalize the two PDFs and submit the three required files](https://github.com/athul1810/click2future/issues/20) | delivery | T18, T19 | Unassigned |

## Task details

<a id="t01"></a>

### T01: Record group details and finalize project scope

Apply the AI6132 technical-review requirements and record the remaining group and submission details.

**Dependencies:** None · **Owner:** Unassigned

- [x] Record Group 7, the four full names, and PRAMOD SANJAY as coordinator from the supplied roster.
- [ ] Confirm the NTULearn cutoff and resolve the difference between the working 14 November date and the course brief's 16 November date.
- [ ] Check NTULearn for additional instructions and use docs/REQUIREMENTS.md for the confirmed minimum requirements.
- [ ] Agree on the Click2Future scope, preserving at least two compared methods and one evaluation dataset; record any change to the five-variant plan.

<a id="t02"></a>

### T02: Verify TC2 login and GPU training setup

Validate the approved TC2 account and fit training to its recorded limits.

**Dependencies:** None · **Owner:** Unassigned

- [x] Record TC2 access approval for Athul and the Slurm limits in docs/COMPUTE.md: 1 GPU, 10 CPUs, 30G system memory per user, 2 running jobs, and a 6-hour job limit.
- [ ] Complete the first login and verify a small scheduled GPU job using the supplied guide and job script; record the GPU model, VRAM, and software versions.
- [ ] Record storage limits, any total usage quota, access expiry, and backup locations; keep credentials outside Git.
- [ ] Define a single-GPU profiling plan and verify checkpoint save/resume across jobs before launching long training runs.

<a id="t03"></a>

### T03: Create a reproducible development environment

Make the CPU development environment reproducible and document training setup on TC2.

**Progress:** The locked CPU environment, diagnostics, artifact conventions, and module commands are implemented. pip is retained in the development lock, and both editor package-list commands are verified. Cluster validation remains open. Included in PR #25.

**Dependencies:** None · **Owner:** Unassigned

- [x] Add a dependency manifest and tested setup instructions with exact versions.
- [ ] Add an environment/device check; verify CPU setup independently and validate TC2 setup after access details are recorded in T02.
- [x] Select artifact locations and a configuration format; keep credentials out of Git.
- [x] Document working commands as modules are implemented; mark unavailable commands clearly.

<a id="t04"></a>

### T04: Freeze simulator, action, and dataset interfaces

Let simulator, model, and evaluation work proceed against one agreed contract.

**Progress:** PR #24 merged the frozen simulator/action contract, versioned schema, condition/group identities, and tiny scene fixture. These unblock downstream implementation; the remaining course/scope decisions stay in T01.

**Dependencies:** T01 · **Owner:** PRAMOD SANJAY

- [x] Finalize units, axis conventions, force duration, friction law, frame timing, and collision event semantics.
- [x] Finalize tensor shapes, timestamps, scene replay metadata, object identity/padding, and conditioning inputs in docs/ARCHITECTURE.md.
- [x] Define scene_group_id and condition_id, including identical-first-frame uncertainty groups.
- [x] Provide a small fixture or schema example that downstream components can consume.

<a id="t05"></a>

### T05: Implement the seeded 64px tabletop simulator

Generate repeatable tabletop clips with usable physical labels.

**Progress:** PR #24 merged seeded circles/rectangles, centre-force rollout labels, and verified replay/control cases. Appearance currently varies solid colours only; textured rendering remains outstanding.

**Dependencies:** T03, T04 · **Owner:** PRAMOD SANJAY

- [ ] Render 2-4 objects with randomized supported shapes/textures, mass, and friction.
- [x] Apply the agreed centre force and export positions, masks, object collisions, and wall contacts.
- [x] Verify deterministic replay, zero-force behavior, force reversal, and a known collision case.
- [x] Replay a saved scene using its specification, config, assets, physics, and action; provide a documented generation/replay command and tolerance.

<a id="t06"></a>

### T06: Build versioned datasets and leakage-safe splits

Produce a traceable dataset with independent visible-scene groups and controlled OOD tests.

**Progress:** PR #24 merged versioned generation, scene-group ID splits, shared-condition physics variants, and hashed manifests. Separate shape/texture/force OOD suites and a measured generation/storage budget remain outstanding.

**Dependencies:** T05 · **Owner:** PRAMOD SANJAY

- [ ] Export a smoke dataset, measure generation cost/storage, and propose the main dataset size.
- [x] Split by scene_group_id; keep all actions and physics variants of a visible scene together.
- [ ] Create train, validation, ID-test, shape-OOD, texture-OOD, force-OOD, and repeated-hidden-physics test manifests with disjoint scene groups.
- [x] Check identical initial inputs within condition_id and independence of hidden physics from appearance.
- [x] Record dataset/schema versions and split hashes; provide procedural generation code, configs, and seeds so the submitted code needs no packaged dataset or external data link.

<a id="t07"></a>

### T07: Implement and validate the evaluation pipeline

Measure generated video behavior consistently across models.

**Progress:** Implementation and simulator validation are ready for review. The active line-of-centres contact rule reaches event F1 0.605 and recall 0.591 versus 0.258 and 0.182 for the previous rule on the same 200 fresh clips. Tracking covers all 18,538 observations. Untrained CVAE output verifies missing-track handling. Contact estimates remain approximate; trained-generator/OOD validation continues with the experiments.

**Dependencies:** T04, T05 · **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA

- [x] Implement ADE/FDE, action-direction and zero-force controls, interval collision accuracy, event precision/recall/F1, and image/temporal/diversity metrics.
- [x] Validate the object tracker and contact estimator on held-out simulator clips; record extraction failures.
- [x] Test metrics on exact, shifted, static, and corrupted controls with expected outcomes.
- [x] Separate video-derived metrics from auxiliary-head metrics and export machine-readable records.
- [x] Define whole-sample best-of-K selection, metric preprocessing, missing-track handling, and collision-aware temporal checks before evaluating models.

<a id="t08"></a>

### T08: Train and validate the shared frame autoencoder

Obtain a compact latent representation that preserves object geometry.

**Dependencies:** T03, T06 · **Owner:** MAITI SHRIJEET

- [ ] Implement or select a permitted autoencoder and record architecture/weight provenance.
- [ ] Fit weights and normalization on training data only; select the checkpoint on validation data and inspect test/OOD reconstruction only after freezing it.
- [ ] Record memory, throughput, latent shapes, and a checkpoint with checksum.
- [ ] Freeze and share the selected autoencoder across diffusion comparisons.

<a id="t09"></a>

### T09: Train the conditional VAE baseline E0

Establish a lower-cost generative baseline with the agreed action inputs.

**Progress:** A 574,659-parameter conditional video VAE, learned prior/posterior, future-only L1 plus KL with warmup, TC2-only trainer, checkpoint/resume, and four-seed prior sampler are implemented. CPU tests cover gradients, no-future-input inference, dataset splits, checkpoint transport, and exported video format. TC2 tiny-batch overfitting, GPU resume, trained samples, and measured budgets remain pending.

**Dependencies:** T03, T06 · **Owner:** Unassigned

- [x] Condition on the first frame, selected-object mask, and numeric force.
- [x] Define the posterior, prior, reconstruction loss, and KL weight; sample from the prior without future frames at inference.
- [ ] Pass a tiny-batch overfit check and produce four seeded samples per condition.
- [ ] Log validation results, model size, training cost, and checkpoint selection.
- [ ] Add a run manifest and qualitative successes/failures.

<a id="t10"></a>

### T10: Implement first-frame and numeric-action diffusion E1/E2

Create the common diffusion backbone and the main numeric-action baseline.

**Progress:** The untrained prototype supports first-frame and numeric conditioning with forward/backward checks. Dataset training, the diffusion schedule/sampler, overfitting, and trained samples remain outstanding.

**Dependencies:** T08 · **Owner:** MAITI SHRIJEET

- [ ] Implement latent video training and sampling with a shared backbone/configuration.
- [ ] Support first-frame-only E1 and selected-mask-plus-numeric-force E2.
- [ ] Pass forward/backward, tiny-batch overfit, and checkpoint resume checks.
- [ ] Measure 64px peak VRAM, seconds/step, and four-sample inference latency.
- [ ] Record baseline samples and proposed run budgets.

<a id="t11"></a>

### T11: Implement object-centric spatial conditioning E3

Test whether localizing the action improves physical action adherence.

**Progress:** Numeric/spatial conditioning switches exist in the prototype with equal parameter counts. Working E3 training/sampling and learned action-control checks remain outstanding.

**Dependencies:** T10, T04 · **Owner:** MAITI SHRIJEET

- [ ] Fuse a selected-object feature, numeric force embedding, and spatial action map into the denoiser.
- [ ] Keep target identification available in E2 so E2-to-E3 isolates the spatial module.
- [ ] Check selected-object changes, zero/reversed action, and empty or invalid masks with fixed sampling seeds.
- [ ] Record added parameter count, ablation switch, and working E3 training/sampling.

<a id="t12"></a>

### T12: Add trajectory and collision supervision E4

Test whether physical supervision improves decoded future behavior.

**Progress:** The shared batch adapter and joint epsilon/auxiliary objective are implemented without changing the denoiser. Real-loader gradients reach the noise output, both auxiliary heads, and shared features, including a positive collision fixture. Head export restores scene IDs, validates feature timesteps, and writes separate auxiliary_head metrics. Compatibility with B's working DDIM sampler is checked, including actual final-head capture. Review/integration of that branch, E4 overfitting, and measured head-versus-video results remain pending.

**Dependencies:** T11, T07 · **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA

- [x] Predict trajectories and object-pair collision probabilities from denoiser features.
- [x] Define timestep supervision, object-slot matching, valid masks, imbalance handling, and loss weights.
- [x] Keep oracle trajectories/collisions/physics out of inference conditioning; any extra object extraction must use first-frame pixels only.
- [ ] Validate loss gradients and tiny-batch behavior; demonstrate E4 training and sampling.
- [ ] Report head predictions separately from generated-video tracking metrics.

<a id="t13"></a>

### T13: Freeze experiment settings and decide the 128px gate

Commit to a comparable experiment suite that fits the remaining budget.

**Dependencies:** T07, T09, T10, T11, T12 · **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA

- [ ] Freeze dataset versions, model controls, sampler, K, metrics, seeds, and checkpoint criteria.
- [ ] Use measured costs to allocate training/evaluation hours and retry reserve.
- [ ] Record a justified proceed/defer decision for 128px; preserve core 64px comparisons.
- [ ] Record all selected settings and validation-only tuning decisions.
- [ ] Use independent runs from a common initialization policy for the main diffusion comparisons; report inherited training separately for warm-started runs.
- [ ] Reserve a small validation-only hyperparameter study, such as sampling-step counts, and record the tested values and compute budget.

<a id="t14"></a>

### T14: Run and analyze the core E0-E4 comparisons

Produce the main evidence for conditioning and supervision claims.

**Progress:** CSV/JSON comparison export is implemented and checked on 24 simulator-control reports from four independent scenes. It retains run/checkpoint IDs, K, missing counts, and scene-group intervals. Actual E0-E4 runs, cost measurements, hyperparameter study, and analysis still depend on T13.

**Dependencies:** T13 · **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA

- [ ] Run the five variants under the frozen protocol and record actual budgets/seed counts.
- [ ] Compare E2-to-E3 and E3-to-E4 with quality, physical behavior, and cost metrics.
- [ ] Report mean and best-of-K trajectory scores separately, plus collision prevalence/F1.
- [ ] Export tables with run IDs, sample counts, group-based uncertainty estimates, and failures.
- [ ] Run the planned hyperparameter study and analyze its quality/cost tradeoff; include strengths, limitations, and visual examples from our own runs.

<a id="t15"></a>

### T15: Evaluate held-out shapes, textures, and forces

Measure which action-conditioned behaviors generalize beyond training.

**Dependencies:** T13 · **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA

- [ ] Evaluate separate OOD suites while controlling the other factors.
- [ ] Use frozen checkpoints and metrics without test-set tuning.
- [ ] Compare against ID performance and include tracker failure rates.
- [ ] Save representative successes/failures and state generalization limits.

<a id="t16"></a>

### T16: Evaluate hidden-physics uncertainty and action controls

Distinguish physically meaningful uncertainty from visual noise.

**Dependencies:** T13 · **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA

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

**Progress:** The source ZIP now includes CVAE training/sampling, the shared E4 objective, and a TC2 job script. The archive still excludes data, weights, links, and working files. Source extraction and replay checks are rerun after changes. Diffusion trainer/sampler, live GPU checkpoint checks, demo integration, and final experiment reproduction remain.

**Dependencies:** T14, T15, T16, T17 · **Owner:** Unassigned

- [ ] Run internal simulator replay, checkpoint loading, and sampling checks; retain all model weights and datasets outside the submission.
- [x] Build group_07_code.zip from an explicit source/config/dependency-file list, with local setup and run instructions.
- [ ] Include procedural data-generation and seed/split logic, model definitions, training, and evaluation source; exclude datasets, model files, external URLs, and cloud-artifact pointers.
- [x] Extract the ZIP in a clean folder and test imports, documented commands, and locally generated smoke inputs without existing datasets or checkpoints.
- [x] Inspect the archive for Git history, environments, caches, logs, credentials, and working documents with links; follow docs/SUBMISSION.md.
- [ ] Keep optional demo recordings and checkpoint-based demo checks internal; document how to regenerate data and train models from the submitted source.

<a id="t20"></a>

### T20: Finalize the two PDFs and submit the three required files

Complete one group submission through NTULearn using the exact required filenames and content restrictions.

**Dependencies:** T18, T19 · **Owner:** Unassigned

- [ ] Export group_07_report.pdf with at least four pages and the required sections/analysis; inspect every page for readable figures, tables, and references.
- [ ] Export group_07_contribution.pdf with every member's full name and reviewed contribution description.
- [ ] Use the confirmed group ID in all three filenames and verify group_07_code.zip contains source code only, with no datasets, model files, or external links.
- [ ] Check both PDFs for external URLs and hyperlinks; include all assessed material directly rather than use cloud-hosted substitutes.
- [ ] Coordinate one group submission of the three files through NTULearn > Assignments > Project 1, using the retained 14 November working deadline until the discrepancy is resolved.
- [ ] Keep the submission receipt privately; if replacing a submission, coordinate the update so the latest three files are consistent. Do not upload slides, checkpoints, or a demo video as extra required files.

The machine-readable initial backlog is [task_backlog.json](task_backlog.json). See [the roadmap](ROADMAP.md) for the working deadline and task dependencies. Intermediate task dates have not been set.

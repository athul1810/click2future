# Task board

Internal completion: **14 November 2026**.

A: Sanjay. B and demo integration: Shrijeet. C: Athul and Ananya jointly.
Ananya is proposed for CVAE; acceptance is pending. Pixel diffusion is adopted;
the optional autoencoder is outside the critical path. See [decisions](DECISIONS.md),
[execution plan](EXECUTION_PLAN_2026-10-16.md) and [CVAE handoff](CVAE_HANDOFF.md).

| ID | Task | Depends on | Owner | Status |
|---|---|---|---|---|
| T01 | [Record group details and finalize project scope](https://github.com/athul1810/click2future/issues/1) | None | Unassigned | In progress |
| T02 | [Verify TC2 login and GPU training setup](https://github.com/athul1810/click2future/issues/2) | None | MAITI SHRIJEET | In progress |
| T03 | [Create a reproducible development environment](https://github.com/athul1810/click2future/issues/3) | None | Unassigned | Done |
| T04 | [Freeze simulator, action, and dataset interfaces](https://github.com/athul1810/click2future/issues/4) | T01 | PRAMOD SANJAY | Done |
| T05 | [Implement the seeded 64px tabletop simulator](https://github.com/athul1810/click2future/issues/5) | T03, T04 | PRAMOD SANJAY | In progress |
| T06 | [Build versioned datasets and leakage-safe splits](https://github.com/athul1810/click2future/issues/6) | T05 | PRAMOD SANJAY | In progress |
| T07 | [Implement and validate the evaluation pipeline](https://github.com/athul1810/click2future/issues/7) | T04, T05 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | In progress |
| T08 | [Optional: train and validate a shared frame autoencoder](https://github.com/athul1810/click2future/issues/8) | T03, T06 | MAITI SHRIJEET | Deferred |
| T09 | [Train the conditional VAE baseline E0](https://github.com/athul1810/click2future/issues/9) | T03, T06 | Unassigned | In progress |
| T10 | [Implement first-frame and numeric-action diffusion E1/E2](https://github.com/athul1810/click2future/issues/10) | T03, T04, T06 | MAITI SHRIJEET | In progress |
| T11 | [Implement object-centric spatial conditioning E3](https://github.com/athul1810/click2future/issues/11) | T10, T04 | MAITI SHRIJEET | In progress |
| T12 | [Add trajectory and collision supervision E4](https://github.com/athul1810/click2future/issues/12) | T11, T07 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | In progress |
| T13 | [Freeze experiment settings and decide the 128px gate](https://github.com/athul1810/click2future/issues/13) | T07, T09, T10, T11 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | In progress |
| T14 | [Run and analyze E0/E2/E3, with gated E1/E4 ablations](https://github.com/athul1810/click2future/issues/14) | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | Not started |
| T15 | [Evaluate held-out shapes, textures, and forces](https://github.com/athul1810/click2future/issues/15) | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | Not started |
| T16 | [Evaluate hidden-physics uncertainty and action controls](https://github.com/athul1810/click2future/issues/16) | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | Not started |
| T17 | [Build the interactive four-future comparison demo](https://github.com/athul1810/click2future/issues/17) | T04, T05, T10 | MAITI SHRIJEET | Not started |
| T18 | [Draft the technical report and contribution statement](https://github.com/athul1810/click2future/issues/18) | T01, T13 | Unassigned | Not started |
| T19 | [Verify reproducibility and prepare the source-code ZIP](https://github.com/athul1810/click2future/issues/19) | T14, T15, T16, T17 | Unassigned | In progress |
| T20 | [Finalize the two PDFs and submit the three required files](https://github.com/athul1810/click2future/issues/20) | T18, T19 | Unassigned | Not started |

## Task details

<a id="t01"></a>

### T01: Record group details and finalize project scope

Apply the AI6132 technical-review requirements and record the remaining group and submission details.

**Dependencies:** None. **Owner:** Unassigned. **Status:** In progress.

- [x] Record Group 7, the four full names, and PRAMOD SANJAY as coordinator from the supplied roster.
- [ ] Confirm the exact NTULearn upload cutoff; preserve the hard 14 November internal completion date regardless of a later upload date.
- [ ] Check NTULearn for additional instructions and use docs/REQUIREMENTS.md for the confirmed minimum requirements.
- [x] Agree on the Click2Future scope, preserving at least two compared methods and one evaluation dataset; record any change to the five-variant plan.

Pixel-first scope adopted by Shrijeet on 16 October. All presentation and submission artifacts must be complete by 14 November. Exact NTULearn upload cutoff remains unverified.

<a id="t02"></a>

### T02: Verify TC2 login and GPU training setup

Validate the approved TC2 account and fit training to its recorded limits.

**Dependencies:** None. **Owner:** MAITI SHRIJEET. **Status:** In progress.

- [x] Record TC2 access approval for Athul and Shrijeet and the Slurm limits in docs/COMPUTE.md: 1 GPU, 10 CPUs, 30G system memory per user, 2 running jobs, and a 6-hour job limit.
- [x] Complete the first login and verify a small scheduled GPU job using the supplied guide and job script; record the GPU model, VRAM, and software versions.
- [ ] Record storage limits, any total usage quota, access expiry, and backup locations; keep credentials outside Git.
- [x] Define a single-GPU profiling plan and verify checkpoint save/resume across jobs before launching long training runs.

A40 setup 42886 and real training jobs 42909/42910 completed. Full model/EMA/optimizer/scheduler/RNG/data-order resume matched the next update with zero tensor difference across allocations. Data-loading-inclusive step time 0.04883 s; four 32-step samples 1.744 s; peak allocated 752,562,176 bytes. No user storage quota or expiry was exposed by the read-only checks. Evidence: docs/results/diffusion_pilot_2026-10-16.json. Generated-video gate failed; this is infrastructure evidence, not model readiness.

<a id="t03"></a>

### T03: Create a reproducible development environment

Make the CPU development environment reproducible and document training setup on TC2.

**Dependencies:** None. **Owner:** Unassigned. **Status:** Done.

- [x] Add a dependency manifest and tested setup instructions with exact versions.
- [x] Add an environment/device check; verify CPU setup independently and validate TC2 setup after access details are recorded in T02.
- [x] Select artifact locations and a configuration format; keep credentials out of Git.
- [x] Document working commands as modules are implemented; mark unavailable commands clearly.

Locked development setup is verified locally and on the TC2 A40 in job 42886. CPU/CUDA diagnostics, dependency checks, tests, artifact conventions and module commands passed. Optional image metrics remain a separate extra. Evidence: docs/results/tc2_probe_2026-10-16.json.

<a id="t04"></a>

### T04: Freeze simulator, action, and dataset interfaces

Let simulator, model, and evaluation work proceed against one agreed contract.

**Dependencies:** T01. **Owner:** PRAMOD SANJAY. **Status:** Done.

- [x] Finalize units, axis conventions, force duration, friction law, frame timing, and collision event semantics.
- [x] Finalize tensor shapes, timestamps, scene replay metadata, object identity/padding, and conditioning inputs in docs/ARCHITECTURE.md.
- [x] Define scene_group_id and condition_id, including identical-first-frame uncertainty groups.
- [x] Provide a small fixture or schema example that downstream components can consume.

PR #24 merged the frozen simulator/action contract, versioned schema, condition/group identities, and tiny scene fixture. These unblock downstream implementation; the remaining course/scope decisions stay in T01.

<a id="t05"></a>

### T05: Implement the seeded 64px tabletop simulator

Generate repeatable tabletop clips with usable physical labels.

**Dependencies:** T03, T04. **Owner:** PRAMOD SANJAY. **Status:** In progress.

- [ ] Render 2-4 objects with randomized supported shapes/textures, mass, and friction.
- [x] Apply the agreed centre force and export positions, masks, object collisions, and wall contacts.
- [x] Verify deterministic replay, zero-force behavior, force reversal, and a known collision case.
- [x] Replay a saved scene using its specification, config, assets, physics, and action; provide a documented generation/replay command and tolerance.

PR #24 merged seeded circles/rectangles, centre-force rollout labels, and verified replay/control cases. Appearance currently varies solid colours only; textured rendering remains outstanding.

<a id="t06"></a>

### T06: Build versioned datasets and leakage-safe splits

Produce a traceable dataset with independent visible-scene groups and controlled OOD tests.

**Dependencies:** T05. **Owner:** PRAMOD SANJAY. **Status:** In progress.

- [x] Export a smoke dataset, measure generation cost/storage, and propose the main dataset size.
- [x] Split by scene_group_id; keep all actions and physics variants of a visible scene together.
- [ ] Create train, validation, ID-test, shape-OOD, texture-OOD, force-OOD, and repeated-hidden-physics test manifests with disjoint scene groups.
- [x] Check identical initial inputs within condition_id and independence of hidden physics from appearance.
- [x] Record dataset/schema versions and split hashes; provide procedural generation code, configs, and seeds so the submitted code needs no packaged dataset or external data link.

Dataset-0.2 generation is merged in PR #31: six controlled actions and 21 clips per visible scene, with all actions and hidden-physics variants in one split. The recommended size is 200 groups and 4,200 clips; final training data is not frozen here. All 84 independently generated pilot clips replayed exactly. The earlier 80-clip dataset-0.1 CPU/storage benchmark remains in the evidence record. Separate shape, texture and force OOD suites remain outstanding.

<a id="t07"></a>

### T07: Implement and validate the evaluation pipeline

Measure generated video behavior consistently across models.

**Dependencies:** T04, T05. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** In progress.

- [x] Implement ADE/FDE, action-direction and zero-force controls, interval collision accuracy, event precision/recall/F1, and image/temporal/diversity metrics.
- [x] Validate the object tracker and contact estimator on held-out simulator clips; record extraction failures.
- [x] Test metrics on exact, shifted, static, and corrupted controls with expected outcomes.
- [x] Separate video-derived metrics from auxiliary-head metrics and export machine-readable records.
- [x] Define whole-sample best-of-K selection, metric preprocessing, missing-track handling, and collision-aware temporal checks before evaluating models.

Evaluation implementation is ready for review in PR #30. The random-scene contact rule reaches event F1 0.605 on its 200-clip check. Controlled scenes expose weaker timing: a fresh 63-clip, three-group check gives event F1 0.551, with all 3,906 object observations tracked. Separate collision-occurrence scoring reaches TP=23, FP=0, FN=1, TN=39 there; it does not establish correct timing. Trained-model and OOD validation remain part of the experiments.

<a id="t08"></a>

### T08: Optional: train and validate a shared frame autoencoder

Obtain a compact latent representation that preserves object geometry.

**Dependencies:** T03, T06. **Owner:** MAITI SHRIJEET. **Status:** Deferred.

- [ ] Implement or select a permitted autoencoder and record architecture/weight provenance.
- [ ] Fit weights and normalization on training data only; select the checkpoint on validation data and inspect test/OOD reconstruction only after freezing it.
- [ ] Record memory, throughput, latent shapes, and a checkpoint with checksum.
- [ ] Freeze and share the selected autoencoder across diffusion comparisons.

Closed as not planned for the adopted pixel-diffusion scope in D011. No autoencoder has been implemented or trained. Reopen only after working 64px samples if measured memory or throughput justifies latents.

<a id="t09"></a>

### T09: Train the conditional VAE baseline E0

Establish a lower-cost generative baseline with the agreed action inputs.

**Dependencies:** T03, T06. **Owner:** Unassigned. **Status:** In progress.

- [x] Condition on the first frame, selected-object mask, and numeric force.
- [x] Define the posterior, prior, reconstruction loss, and KL weight; sample from the prior without future frames at inference.
- [ ] Pass a tiny-batch overfit check and produce four seeded samples per condition.
- [ ] Log validation results, model size, training cost, and checkpoint selection.
- [ ] Add a run manifest and qualitative successes/failures.

A 574,659-parameter conditional video VAE, learned prior/posterior, future-only L1 plus KL with warmup, TC2-only trainer, checkpoint/resume, and four-seed prior sampler are implemented. Both dataset-0.1 and dataset-0.2 work through the same checked split index. A native 1.1 wall-force condition passes loader normalization and four-video untrained prior export. CPU checks do not establish model quality. TC2 tiny-batch overfitting, GPU resume, trained samples, and measured budgets remain pending.

<a id="t10"></a>

### T10: Implement first-frame and numeric-action diffusion E1/E2

Create the common diffusion backbone and the main numeric-action baseline.

**Dependencies:** T03, T04, T06. **Owner:** MAITI SHRIJEET. **Status:** In progress.

- [x] Implement pixel video training and DDIM sampling with a shared backbone/configuration.
- [x] Support first-frame-only E1 and selected-mask-plus-numeric-force E2.
- [ ] Pass forward/backward, tiny-batch overfit, and checkpoint resume checks.
- [x] Measure 64px peak VRAM, seconds/step, and four-sample inference latency.
- [x] Record baseline samples and proposed run budgets.

Trainer, sampler, EMA and full checkpoint implemented. Exact CPU continuation passes with/without auxiliary losses; cross-job A40 continuation matched with zero tensor difference in jobs 42909/42910. Four-clip, 2000-step pilot reduced fixed training-noise MSE from 1.02554 to 0.01965, but generated clips remain noisy and 0/248 future object observations were trackable. Tiny-overfit and held-out action-response gates remain open. Follow-up jobs 42917 (10000-step epsilon) and 42920 (2000-step velocity) completed. Longer epsilon still failed tracking; velocity reduced visible noise but recovered only 1-7/62 observations. Tiny-overfit remains open. Next isolate a deterministic single clip, noise-level reconstruction and precision. Evidence: docs/results/diffusion_debug_2026-10-16.json.

<a id="t11"></a>

### T11: Implement object-centric spatial conditioning E3

Test whether localizing the action improves physical action adherence.

**Dependencies:** T10, T04. **Owner:** MAITI SHRIJEET. **Status:** In progress.

- [x] Fuse a selected-object feature, numeric force embedding, and spatial action map into the denoiser.
- [x] Keep target identification available in E2 so E2-to-E3 isolates the spatial module.
- [ ] Check selected-object changes, zero/reversed action, and empty or invalid masks with fixed sampling seeds.
- [ ] Record added parameter count, ablation switch, and working E3 training/sampling.

Numeric/spatial conditioning and selected-object identity are implemented with identical backbone capacity. Forward/backward and zero-force wiring controls pass. The trainer/sampler supports both modes. Trained selected-object and reversed-force controls, coherent E3 videos and the matched E2/E3 comparison remain open.

<a id="t12"></a>

### T12: Add trajectory and collision supervision E4

Test whether physical supervision improves decoded future behavior.

**Dependencies:** T11, T07. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** In progress.

- [x] Predict trajectories and object-pair collision probabilities from denoiser features.
- [x] Define timestep supervision, object-slot matching, valid masks, imbalance handling, and loss weights.
- [x] Keep oracle trajectories/collisions/physics out of inference conditioning; any extra object extraction must use first-frame pixels only.
- [ ] Validate loss gradients and tiny-batch behavior; demonstrate E4 training and sampling.
- [x] Report head predictions separately from generated-video tracking metrics.

Canonical trajectory/contact losses, real-loader gradients and separate auxiliary-head export are implemented. Actual final DDIM heads are captured for both prediction targets without changing sampled pixels or RNG state. E4 overfitting and trained head-versus-video comparisons remain open.

<a id="t13"></a>

### T13: Freeze experiment settings and decide the 128px gate

Commit to a comparable experiment suite that fits the remaining budget.

**Dependencies:** T07, T09, T10, T11. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** In progress.

- [ ] Freeze dataset versions, model controls, sampler, K, metrics, seeds, and checkpoint criteria.
- [ ] Use measured costs to allocate training/evaluation hours and retry reserve.
- [x] Record a justified proceed/defer decision for 128px; preserve core 64px comparisons.
- [ ] Record all selected settings and validation-only tuning decisions.
- [ ] Use independent runs from a common initialization policy for the main diffusion comparisons; report inherited training separately for warm-started runs.
- [x] Reserve a small validation-only hyperparameter study, such as sampling-step counts, and record the tested values and compute budget.

Recorded 64px scope, deferred 128px, common sampling/training seeds, independent initialization policy, extractor settings and bootstrap rules in configs/evaluation/comparison.yaml. Reserved a validation-only 16/32/64-step study on at most 16 conditions with a 0.5 GPU-hour cap. The batch evaluator hashes all cases, references, metric code and configs. Final data/OOD versions, successful model settings, checkpoints and main training budgets are not frozen.

<a id="t14"></a>

### T14: Run and analyze E0/E2/E3, with gated E1/E4 ablations

Produce the main evidence for conditioning and supervision claims.

**Dependencies:** T13. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** Not started.

- [ ] Run E0/E2/E3 under the frozen protocol; add E1/E4 only after explicit budget gates and record actual budgets/seed counts.
- [ ] Compare CVAE-to-diffusion and E2-to-E3; compare E3-to-E4 if funded, using generated-video quality, physical behavior and cost.
- [ ] Report mean and best-of-K trajectory scores separately, plus collision prevalence/F1.
- [ ] Export tables with run IDs, sample counts, group-based uncertainty estimates, and failures.
- [ ] Run the planned hyperparameter study and analyze its quality/cost tradeoff; include strengths, limitations, and visual examples from our own runs.

The batch evaluator produces trajectory, image and uncertainty reports plus scene-group comparison tables for every frozen case. Full-length oracle/static simulator controls run end to end with SSIM and LPIPS. Actual trained E0/E2/E3 runs, hyperparameter results and their comparisons remain pending.

<a id="t15"></a>

### T15: Evaluate held-out shapes, textures, and forces

Measure which action-conditioned behaviors generalize beyond training.

**Dependencies:** T13. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** Not started.

- [ ] Evaluate separate OOD suites while controlling the other factors.
- [ ] Use frozen checkpoints and metrics without test-set tuning.
- [ ] Compare against ID performance and include tracker failure rates.
- [ ] Save representative successes/failures and state generalization limits.

The batch evaluator can apply identical frozen extraction/reporting to held-out splits and rejects missing cases or changed inputs. Separate shape/texture/force OOD data from T06 and trained frozen checkpoints remain required.

<a id="t16"></a>

### T16: Evaluate hidden-physics uncertainty and action controls

Distinguish physically meaningful uncertainty from visual noise.

**Dependencies:** T13. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** Not started.

- [ ] Sample K=4 model futures for identical visible conditions and compare with multiple simulator rollouts.
- [ ] Measure trajectory spread, reference coverage, sample plausibility, and collision-frequency variation.
- [ ] Evaluate zero force, reversed force, and selected-object changes using the fixed action-control protocol and report stationary/tracking failures.
- [ ] Present quality/adherence alongside diversity and clearly label single-reference metrics.

The batch evaluator groups all hidden-physics references per condition and reports K=4 spread, coverage, plausibility distances and failure counts alongside image/trajectory quality. Full-length simulator controls validate this workflow. Actual generated ensembles and paired action-control measurements remain open.

<a id="t17"></a>

### T17: Build the interactive four-future comparison demo

Make the model and simulator comparison understandable and repeatable.

**Dependencies:** T04, T05, T10. **Owner:** MAITI SHRIJEET. **Status:** Not started.

- [ ] Support preset scenes and synthetic-domain image input with selected-object masks.
- [ ] Map drag arrows to the agreed force convention and generate four seeded futures.
- [ ] Show simulator ground truth only when scene metadata exists.
- [ ] Label the friction slider as simulator-only for hidden-physics checkpoints.
- [ ] Add trajectory overlays, loading/error states, and a documented launch path.

Shared EMA inference API exports evaluator-ready uint8 videos with exact frame 0 and paired sampling seeds. UI and live/cached fallback remain to be implemented.

<a id="t18"></a>

### T18: Draft the technical report and contribution statement

Prepare the two required PDFs using the course structure and the team's measured results.

**Dependencies:** T01, T13. **Owner:** Unassigned. **Status:** Not started.

- [ ] Draft at least four pages using introduction, methods, experiments and analysis, and conclusion; follow docs/REPORT_OUTLINE.md.
- [ ] Compare at least two methods on at least one dataset and explain the implemented methods and related work without unsupported novelty claims.
- [ ] Include method comparison, hyperparameter analysis, component analysis, visualizations, strengths, limitations, and future extensions using our own measured runs.
- [ ] Draft each member's actual contribution using docs/CONTRIBUTION_TEMPLATE.md; collect reviewed descriptions rather than invent shares.
- [ ] Embed assessed figures and tables directly in the report; use bibliography entries without external links in both submitted PDFs.

<a id="t19"></a>

### T19: Verify reproducibility and prepare the source-code ZIP

Prepare a source-only submission package and check that it can be used without the team's working files.

**Dependencies:** T14, T15, T16, T17. **Owner:** Unassigned. **Status:** In progress.

- [ ] Run internal simulator replay, trained-checkpoint loading, and sampling checks; retain all model weights and datasets outside the submission.
- [x] Build group_07_code.zip from an explicit source/config/dependency-file list, with local setup and run instructions.
- [x] Include procedural data-generation and seed/split logic, model definitions, training, and evaluation source; exclude datasets, model files, external URLs, and cloud-artifact pointers.
- [x] Extract the ZIP in a clean folder and test imports, documented commands, and locally generated smoke inputs without existing datasets or checkpoints.
- [x] Inspect the archive for Git history, environments, caches, logs, credentials, and working documents with links; follow docs/SUBMISSION.md.
- [x] Keep optional demo recordings and checkpoint-based demo checks internal; document how to regenerate data and train models from the submitted source.

The source-only ZIP includes diffusion/CVAE training and sampling, both TC2 job scripts, procedural data/splits, all evaluators and standalone instructions. Builder, inventory, URL/content inspection and source extraction checks pass. Actual trained-checkpoint reproduction and the final post-experiment package check remain open.

<a id="t20"></a>

### T20: Finalize the two PDFs and submit the three required files

Complete one group submission through NTULearn using the exact required filenames and content restrictions.

**Dependencies:** T18, T19. **Owner:** Unassigned. **Status:** Not started.

- [ ] Export group_07_report.pdf with at least four pages and the required sections/analysis; inspect every page for readable figures, tables, and references.
- [ ] Export group_07_contribution.pdf with every member's full name and reviewed contribution description.
- [ ] Use the confirmed group ID in all three filenames and verify group_07_code.zip contains source code only, with no datasets, model files, or external links.
- [ ] Check both PDFs for external URLs and hyperlinks; include all assessed material directly rather than use cloud-hosted substitutes.
- [ ] Coordinate one group submission through NTULearn > Assignments > Project 1; all three files must be complete by 14 November regardless of the verified upload cutoff.
- [ ] Keep the submission receipt privately; if replacing a submission, coordinate the update so the latest three files are consistent. Do not upload slides, checkpoints, or a demo video as extra required files.

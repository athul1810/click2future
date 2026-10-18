# Task board

Internal completion: **14 November 2026**.

A: Sanjay. B and demo integration: Shrijeet. C: Athul and Ananya jointly.
Ananya confirmed CVAE on 17 October. Pixel diffusion is adopted;
the optional autoencoder is outside the critical path. See [decisions](DECISIONS.md),
[execution plan](EXECUTION_PLAN_2026-10-16.md) and [CVAE handoff](CVAE_HANDOFF.md).

| ID | Task | Depends on | Owner | Status |
|---|---|---|---|---|
| T01 | [Record group details and finalize project scope](https://github.com/athul1810/click2future/issues/1) | None | Unassigned | Done |
| T02 | [Verify TC2 login and GPU training setup](https://github.com/athul1810/click2future/issues/2) | None | MAITI SHRIJEET | Done |
| T03 | [Create a reproducible development environment](https://github.com/athul1810/click2future/issues/3) | None | Unassigned | Done |
| T04 | [Freeze simulator, action, and dataset interfaces](https://github.com/athul1810/click2future/issues/4) | T01 | PRAMOD SANJAY | Done |
| T05 | [Implement the seeded 64px tabletop simulator](https://github.com/athul1810/click2future/issues/5) | T03, T04 | PRAMOD SANJAY | Done |
| T06 | [Build versioned datasets and leakage-safe splits](https://github.com/athul1810/click2future/issues/6) | T05 | PRAMOD SANJAY | Done |
| T07 | [Implement and validate the evaluation pipeline](https://github.com/athul1810/click2future/issues/7) | T04, T05 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | Done |
| T08 | [Optional: train and validate a shared frame autoencoder](https://github.com/athul1810/click2future/issues/8) | T03, T06 | MAITI SHRIJEET | Deferred |
| T09 | [Train the conditional VAE baseline E0](https://github.com/athul1810/click2future/issues/9) | T03, T06 | KAPOOR ANANYA | Done |
| T10 | [Implement first-frame and numeric-action diffusion E1/E2](https://github.com/athul1810/click2future/issues/10) | T03, T04, T06 | MAITI SHRIJEET | In progress |
| T11 | [Implement object-centric spatial conditioning E3](https://github.com/athul1810/click2future/issues/11) | T10, T04 | MAITI SHRIJEET | In review |
| T12 | [Add trajectory and collision supervision E4](https://github.com/athul1810/click2future/issues/12) | T11, T07 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | In review |
| T13 | [Freeze experiment settings and decide the 128px gate](https://github.com/athul1810/click2future/issues/13) | T07, T09, T10, T11 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | In progress |
| T14 | [Run and analyze E0/E2/E3, with gated E1/E4 ablations](https://github.com/athul1810/click2future/issues/14) | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | In progress |
| T15 | [Evaluate held-out shapes, textures, and forces](https://github.com/athul1810/click2future/issues/15) | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | Not started |
| T16 | [Evaluate hidden-physics uncertainty and action controls](https://github.com/athul1810/click2future/issues/16) | T13 | BOBAN ATHUL KRISHNA and KAPOOR ANANYA | In review |
| T17 | [Build the interactive four-future comparison demo](https://github.com/athul1810/click2future/issues/17) | T04, T05, T10 | MAITI SHRIJEET | In progress |
| T18 | [Draft the technical report and contribution statement](https://github.com/athul1810/click2future/issues/18) | T01, T13 | Unassigned | In progress |
| T19 | [Verify reproducibility and prepare the source-code ZIP](https://github.com/athul1810/click2future/issues/19) | T03, T04 | Unassigned | Done |
| T20 | [Finalize the two PDFs and submit the three required files](https://github.com/athul1810/click2future/issues/20) | T14, T15, T16, T17, T18, T19 | Unassigned | Not started |

## Task details

<a id="t01"></a>

### T01: Record group details and finalize project scope

Apply the AI6132 technical-review requirements and record the remaining group and submission details.

**Dependencies:** None. **Owner:** Unassigned. **Status:** Done.

- [x] Record Group 7, the four full names, and PRAMOD SANJAY as coordinator from the supplied roster.
- [x] Record the course brief requirements, the 14 November 2026 working deadline and the unresolved 16 November upload-date discrepancy.
- [x] Agree on the Click2Future scope, preserving at least two compared methods and one evaluation dataset; record any change to the five-variant plan.

Group 7, the full roster, coordinator, course requirements and adopted 64px E0/E2/E3 scope are recorded. Final NTULearn cutoff and updated-instruction checks are tracked in T20; they are not claimed as verified.

<a id="t02"></a>

### T02: Verify TC2 login and GPU training setup

Validate the approved TC2 account and fit training to its recorded limits.

**Dependencies:** None. **Owner:** MAITI SHRIJEET. **Status:** Done.

- [x] Record TC2 access approval for Athul and Shrijeet and the Slurm limits in docs/COMPUTE.md: 1 GPU, 10 CPUs, 30G system memory per user, 2 running jobs, and a 6-hour job limit.
- [x] Complete the first login and verify a small scheduled GPU job using the supplied guide and job script; record the GPU model, VRAM, and software versions.
- [x] Check available storage/usage/expiry information, record unexposed limits explicitly and verify private checkpoint backups; keep credentials outside Git.
- [x] Define a single-GPU profiling plan and verify checkpoint save/resume across jobs before launching long training runs.

A40 setup 42886 and real training jobs 42909/42910 completed. Full model/EMA/optimizer/scheduler/RNG/data-order resume matched the next update with zero tensor difference across allocations. Data-loading-inclusive step time 0.04883 s; four 32-step samples 1.744 s; peak allocated 752,562,176 bytes. No user storage quota or expiry was exposed by the read-only checks. Evidence: docs/results/diffusion_pilot_2026-10-16.json. Generated-video gate failed; this is infrastructure evidence, not model readiness.

Athul setup job 43158 verified the A40 and locked CUDA environment. Scheduled training, complete-state continuation and ignored local checkpoint backups are verified. mytcinfo confirms 1 GPU, 10 CPUs, 30G RAM and 6 hours per job; a third submission was refused while two jobs were queued/running. Storage quota, total GPU-hours and service expiry are not exposed by the checked metadata and remain explicitly unknown. Model-quality failures stay in T09/T10.

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

**Dependencies:** T03, T04. **Owner:** PRAMOD SANJAY. **Status:** Done.

- [x] Render 2-4 objects with randomized supported shapes/textures, mass, and friction.
- [x] Apply the agreed centre force and export positions, masks, object collisions, and wall contacts.
- [x] Verify deterministic replay, zero-force behavior, force reversal, and a known collision case.
- [x] Replay a saved scene using its specification, config, assets, physics, and action; provide a documented generation/replay command and tolerance.

Textured rendering, object-attached stripe/checker patterns, physical triangles and 2-4-object texture checks are implemented and merged in PR #32. Native centre-force labels and legacy replay remain unchanged. The independent smoke replays 332 new clips plus 80 dataset-0.1 and 84 dataset-0.2 saved clips exactly. No learned-model result is claimed. The merged stack passed 391 tests and four subtests, including real LPIPS, on 17 October.

<a id="t06"></a>

### T06: Build versioned datasets and leakage-safe splits

Produce a traceable dataset with independent visible-scene groups and controlled OOD tests.

**Dependencies:** T05. **Owner:** PRAMOD SANJAY. **Status:** Done.

- [x] Export a smoke dataset, measure generation cost/storage, and propose the main dataset size.
- [x] Split by scene_group_id; keep all actions and physics variants of a visible scene together.
- [x] Create train, validation, ID-test, shape-OOD, texture-OOD, force-OOD, and repeated-hidden-physics test manifests with disjoint scene groups.
- [x] Check identical initial inputs within condition_id and independence of hidden physics from appearance.
- [x] Record dataset/schema versions and split hashes; provide procedural generation code, configs, and seeds so the submitted code needs no packaged dataset or external data link.

Dataset-0.2 remains available through merged PR #31. Separate dataset-0.3 adds train, validation, ID, shape-OOD, texture-OOD, force-OOD and repeated-physics splits with disjoint visible seeds and complete reference coverage. A 332-clip native smoke took 3.176 seconds to generate and occupies 3,129,140 bytes; every clip replays exactly. Models can load the new corpus, but its final size and training revision are not frozen. Implementation merged in PR #32; trained generalization remains T15. The merged stack passed 391 tests and four subtests, including real LPIPS, on 17 October.

<a id="t07"></a>

### T07: Implement and validate the evaluation pipeline

Measure generated video behavior consistently across models.

**Dependencies:** T04, T05. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** Done.

- [x] Implement ADE/FDE, action-direction and zero-force controls, interval collision accuracy, event precision/recall/F1, and image/temporal/diversity metrics.
- [x] Validate the object tracker and contact estimator on held-out simulator clips; record extraction failures.
- [x] Test metrics on exact, shifted, static, and corrupted controls with expected outcomes.
- [x] Separate video-derived metrics from auxiliary-head metrics and export machine-readable records.
- [x] Define whole-sample best-of-K selection, metric preprocessing, missing-track handling, and collision-aware temporal checks before evaluating models.

Evaluation implementation merged in PR #30. The random-scene contact rule reaches event F1 0.605 on its 200-clip check. Controlled scenes expose weaker timing: a fresh 63-clip, three-group check gives event F1 0.551, with all 3,906 object observations tracked. Separate collision-occurrence scoring reaches TP=23, FP=0, FN=1, TN=39 there; it does not establish correct timing. Trained-model and OOD validation remain part of the experiments. The merged stack passed 391 tests and four subtests, including real LPIPS, on 17 October.

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

**Dependencies:** T03, T06. **Owner:** KAPOOR ANANYA. **Status:** Done.

- [x] Condition on the first frame, selected-object mask, and numeric force.
- [x] Define the posterior, prior, reconstruction loss, and KL weight; sample from the prior without future frames at inference.
- [x] Pass a tiny-batch overfit check and produce four seeded samples per condition.
- [x] Log validation results, model size, training cost, and checkpoint selection.
- [x] Add a run manifest and qualitative successes/failures.

PR #37 merged after independent review. Boundary-weighted CVAE training in job 43199 passes the declared selected-object tiny-training gate: 244/248 tracked observations, selected ADE 0.00249 and posterior MAE 0.00392 across four seeded prior samples. All-object completeness remains 0/4 and validation tracking 0/248; full-data training and held-out quality remain open under T13-T16. Ananya confirmed ownership on 17 October; Athul authored the implementation and these diagnostics. Evidence: results/foreground-loss-checks.json.

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

A separate 5000-step single-clip FP32 velocity diagnostic completed in job 43168. Best EMA sampling tracked 109/248 future observations at 32 steps and failed the unchanged quality gate. Noise-level reconstruction separates object/background errors and precision; this joint one-clip/FP32 change does not isolate a precision effect. Evidence: results/diffusion-single-clip-checks.json.

Candidate tracker-v2 update: the saved #37 boundary-loss EMA/32-step RGB
spatial diagnostic tracks 247/248 observations, with all four selected tracks
complete and selected ADE 0.00348. This passes the original selected-object
training gate under v2. Version 2 is still awaiting Ananya's calibration review;
the shared acceptance box stays open until that review. Shrijeet's separate
#39/#41 structured v2 rescore and video hashes are now recorded in
results/diffusion-tracker-v2-checks.json and were reviewed in PR #45. Preserve all historical v1 scores.
The multi-scene E2/E3 validation failures remain separate and unresolved.

<a id="t11"></a>

### T11: Implement object-centric spatial conditioning E3

Test whether localizing the action improves physical action adherence.

**Dependencies:** T10, T04. **Owner:** MAITI SHRIJEET. **Status:** In review.

- [x] Fuse a selected-object feature, numeric force embedding, and spatial action map into the denoiser.
- [x] Keep target identification available in E2 so E2-to-E3 isolates the spatial module.
- [x] Check selected-object changes, zero/reversed action, and empty or invalid masks with fixed sampling seeds.
- [x] Record added parameter count, ablation switch, and working E3 training/sampling.

The E3 implementation checks are complete and in review. E2 and E3 each have 964,035 parameters; spatial routing adds no capacity. The two actual 5,000-update checkpoints respond to changed selection, zero force and reversal under one fixed-noise input, and empty/nonbinary/wrong-shape masks are rejected before inference. The existing 96-video validation and 16-pair selected-object controls have also run for both methods. Their generated-video action quality fails through missing tracks, so no physical action-adherence improvement is claimed. Model quality and final comparisons remain in T14-T16. Evidence: results/conditioning-implementation-checks.json and results/multiscene-validation-checks.json. This implementation issue closes when its reviewed changes reach main.

<a id="t12"></a>

### T12: Add trajectory and collision supervision E4

Test whether physical supervision improves decoded future behavior.

**Dependencies:** T11, T07. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** In review.

- [x] Predict trajectories and object-pair collision probabilities from denoiser features.
- [x] Define timestep supervision, object-slot matching, valid masks, imbalance handling, and loss weights.
- [x] Keep oracle trajectories/collisions/physics out of inference conditioning; any extra object extraction must use first-frame pixels only.
- [x] Validate loss gradients and tiny-batch behavior; demonstrate E4 training and sampling.
- [x] Report head predictions separately from generated-video tracking metrics.

E4 training, sampling and actual final-head capture are measured in PR #43. The 5,000-update multi-scene run has head ADE 0.1314 and event F1 0.1324, while RGB tracks 162/5,952 observations. The separate collision overfit check learns the contact (head event F1 1.0 on its single training clip) and tracks 247/248 RGB observations. Its one missing selected observation still fails the strict gate. Implementation is ready for review; generated-video quality remains in T14/T16. Evidence: results/multiscene-validation-checks.json.

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

The bounded pilot freezes dataset-0.3 seed 730000, 16 training/four validation scenes, all four model configs, checkpoint criteria, sampler seeds and candidate tracker version 2. It uses a combined 1.5 allocated GPU-hour cap. The 16/32/64-step study is complete. The original four methods fail useful validation motion. E5 and three E6 follow-ups are measured separately; E6 improves stationarity but still fails contact quality and has incomplete wall tracks. No final model/sampler or main test protocol is selected. No 128px or E1 run was added. See MULTISCENE_PILOT.md and results/multiscene-validation-checks.json.

separate e5 follow-up: one seed, 6000 training updates with step 5000 selected by validation denoising loss. pose supervision changes the method; final protocol and independent repeats remain open. see docs/TRAJECTORY_PILOT.md.

<a id="t14"></a>

### T14: Run and analyze E0/E2/E3, with gated E1/E4 ablations

Produce the main evidence for conditioning and supervision claims.

**Dependencies:** T13. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** In progress.

- [ ] Run E0/E2/E3 under the frozen protocol; add E1/E4 only after explicit budget gates and record actual budgets/seed counts.
- [ ] Compare CVAE-to-diffusion and E2-to-E3; compare E3-to-E4 if funded, using generated-video quality, physical behavior and cost.
- [ ] Report mean and best-of-K trajectory scores separately, plus collision prevalence/F1.
- [x] Export tables with run IDs, sample counts, group-based uncertainty estimates, and failures.
- [x] Run the planned hyperparameter study and analyze its quality/cost tradeoff; include strengths, limitations, and visual examples from our own runs.

The four-method TC2 pilot and validation sampler study are measured in PR #43. Each method generates 96 validation videos and 32 selected-object-control videos. Tables retain SSIM/LPIPS, scene bootstrap intervals, null ADE/FDE/coverage, contact prevalence and tracking/action failures. All four original methods fail; static copying has better appearance scores. Flow follow-ups are exploratory and stay separate. The separate E5/E6 follow-ups now include a reviewed contact/stationarity comparison and independently recomputed action-family uncertainty. These combined training changes do not replace the matched E2/E3 question. Final useful-checkpoint comparisons and independent repeats remain open. Evidence: results/multiscene-validation-checks.json; results/contact-uncertainty-comparison.json; report/main.tex.

<a id="t15"></a>

### T15: Evaluate held-out shapes, textures, and forces

Measure which action-conditioned behaviors generalize beyond training.

**Dependencies:** T13. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** Not started.

- [ ] Evaluate separate OOD suites while controlling the other factors.
- [ ] Use frozen checkpoints and metrics without test-set tuning.
- [ ] Compare against ID performance and include tracker failure rates.
- [ ] Save representative successes/failures and state generalization limits.

Separate dataset-0.3 ID, shape, texture, force and physics suites are implemented and merged. The multi-scene pilot is trained on that corpus, but validation generation and action response fail. All 248 test clips remain reserved; there is no trained ID/OOD result. Improve validation before opening these suites. Extractor failures and candidate tracker version 2 remain explicit in MULTISCENE_PILOT.md.

<a id="t16"></a>

### T16: Evaluate hidden-physics uncertainty and action controls

Distinguish physically meaningful uncertainty from visual noise.

**Dependencies:** T13. **Owner:** BOBAN ATHUL KRISHNA and KAPOOR ANANYA. **Status:** In review.

- [x] Sample K=4 model futures for identical visible conditions and compare with multiple simulator rollouts.
- [x] Measure trajectory spread, reference coverage, sample plausibility, and collision-frequency variation.
- [x] Evaluate zero force, reversed force, and selected-object changes using the fixed action-control protocol and report stationary/tracking failures.
- [x] Present quality/adherence alongside diversity and clearly label single-reference metrics.

The validation measurement work is complete and ready for review in #47, stacked on #43 and #42. The frozen E5 audit finds penetration in 22/96 clips versus 0/84 oracle controls and full-horizon zero-force drift in 16/16 clips. PR #48 adds three E6 packages. The paired-action package has 0/16 zero-force drift, 0/32 passive drift and penetration in 12/96 clips. All six incomplete clips are wall cases. The reproducible clipping audit traces all 32 missing observations to sprite area lost at the canvas. Independent C review reproduces all 72 E6 future/uncertainty reports and all 18 action strata. Forward spread remains below the finite references and wall coverage is undefined. Version 2 passive scores retain the original reports and count exactly one pixel as drift. Overall physics quality, independent repeats and final ID/OOD results remain open. See CONTACT_UNCERTAINTY_AUDIT.md and results/contact-uncertainty-comparison.json.


<a id="t17"></a>

### T17: Build the interactive four-future comparison demo

Make the model and simulator comparison understandable and repeatable.

**Dependencies:** T04, T05, T10. **Owner:** MAITI SHRIJEET. **Status:** In progress.

- [x] Support declared development/validation presets with frame-0 selected-object masks; image upload is deferred.
- [x] Map drag arrows to positive-x-right/positive-y-down forces and generate four paired seeded futures with the offline sampler.
- [x] Keep the labelled simulator reference and hidden physics outside neural inference.
- [x] Provide synchronized playback, bounded generation, cancellation, visible pixel-track failures, labelled exact replay and a documented local launch path.
- [ ] Pass held-out motion/action quality and rehearse the final demo offline; repeat latency measurements on the final presentation setup.

The app now supports E6 with the shared offline sampler, generated-pixel overlays and missing counts. PR #48 is reviewed and merged into #47. The source ZIP with the clipping diagnostic passes 550 tests plus four subtests; the unchanged E6-grid/E5 interface passed all 30 browser checks during PR #48 review. Four primary futures fit laptop screens, comparison is explicit and reduced-motion playback stays paused. The earlier five-request E5/E3 latency estimate of 5.31s median and 5.36s empirical p95 belongs to the RTX 3060 laptop; it does not benchmark the new E6/E5 setup. Final physical quality, fresh final-model timing and an offline presentation rehearsal remain open. Candidate tracker v2 awaits independent review. See app/README.md and CONTACT_TRAINING.md.


<a id="t18"></a>

### T18: Draft the technical report and contribution statement

Prepare the two required PDFs using the course structure and the team's measured results.

**Dependencies:** T01, T13. **Owner:** Unassigned. **Status:** In progress.

- [x] Draft at least four pages using introduction, methods, experiments and analysis, and conclusion; follow docs/REPORT_OUTLINE.md.
- [x] Compare at least two methods on at least one dataset and explain the implemented methods and related work without unsupported novelty claims.
- [x] Include method comparison, hyperparameter analysis, component analysis, visualizations, strengths, limitations, and future extensions using our own measured runs.
- [ ] Draft each member's actual contribution using docs/CONTRIBUTION_TEMPLATE.md; collect reviewed descriptions rather than invent shares.
- [ ] Embed assessed figures and tables directly in the report; use bibliography entries without external links in both submitted PDFs.

The fifteen-page working draft retains the earlier method comparison, sampler study and CVAE follow-ups, and now includes E5, all three E6 packages, the corrected passive scores, action-family uncertainty and a fixed E5/E6 four-seed figure. All 15 pages are rendered and inspected, with no external URLs or hyperlink annotations. report/contributions.md records evidence-supported draft descriptions; member reviews and Ananya's completed-work account remain pending. Final export remains open. Source: report/main.tex; report/evidence.json. This is not the final submission PDF.

<a id="t19"></a>

### T19: Verify reproducibility and prepare the source-code ZIP

Prepare a source-only submission package and check that it can be used without the team's working files.

**Dependencies:** T03, T04. **Owner:** Unassigned. **Status:** Done.

- [x] Run internal simulator replay, trained-checkpoint loading, and sampling checks; retain all model weights and datasets outside the submission.
- [x] Build group_07_code.zip from an explicit source/config/dependency-file list, with local setup and run instructions.
- [x] Include procedural data-generation and seed/split logic, model definitions, training, and evaluation source; exclude datasets, model files, external URLs, and cloud-artifact pointers.
- [x] Extract the ZIP in a clean folder and test imports, documented commands, and locally generated smoke inputs without existing datasets or checkpoints.
- [x] Inspect the archive for Git history, environments, caches, logs, credentials, and working documents with links; follow docs/SUBMISSION.md.
- [x] Keep optional demo recordings and checkpoint-based demo checks internal; document how to regenerate data and train models from the submitted source.

Source packaging and trained-checkpoint reproduction are verified. TC2 job 43174 ran real 5000-step E0/E3 checkpoints through the frozen sampler and all three metric families over the complete one-condition validation split, producing eight videos. Model quality failed and stays in T09/T10/T14-T16. Source installation, replay, archive inspection and all 364 tests plus four subtests pass. Implementation merged after independent review; T20 retains final source regeneration and post-experiment artifact checks. The merged stack passed 391 tests and four subtests, including real LPIPS, on 17 October. Evidence: `results/trained-checkpoint-suite-checks.json`.

<a id="t20"></a>

### T20: Finalize the two PDFs and submit the three required files

Complete one group submission through NTULearn using the exact required filenames and content restrictions.

**Dependencies:** T14, T15, T16, T17, T18, T19. **Owner:** Unassigned. **Status:** Not started.

- [ ] Confirm the live NTULearn upload cutoff while preserving 14 November 2026 as the working deadline; do not assume the PDF resolves the discrepancy.
- [ ] Check NTULearn for any updated submission instructions before exporting and uploading the final artifacts.

- [ ] Export group_07_report.pdf with at least four pages and the required sections/analysis; inspect every page for readable figures, tables, and references.
- [ ] Export group_07_contribution.pdf with every member's full name and reviewed contribution description.
- [ ] Use the confirmed group ID in all three filenames and verify group_07_code.zip contains source code only, with no datasets, model files, or external links. Regenerate and verify this final archive after all experiment/demo changes.
- [ ] Check both PDFs for external URLs and hyperlinks; include all assessed material directly rather than use cloud-hosted substitutes.
- [ ] Coordinate one group submission through NTULearn > Assignments > Project 1; all three files must be complete by 14 November regardless of the verified upload cutoff.
- [ ] Keep the submission receipt privately; if replacing a submission, coordinate the update so the latest three files are consistent. Do not upload slides, checkpoints, or a demo video as extra required files.

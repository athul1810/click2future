# Decision log

## D001: Initial technical scope: 7 October 2026

- Compare a conditional VAE and video diffusion for action-conditioned tabletop futures.
- Test spatial action conditioning and auxiliary trajectory/collision supervision separately.
- Start at 64px; consider 128px after profiling on TC2.

## D004: Simulator and sample contract: 15 October 2026

- Use Pymunk 7.3.1 for a deterministic 1 × 1, zero-gravity, top-down world.
- Run physics at 240 Hz with 10 solver iterations and render 32 frames at 8 FPS.
- Apply one centre force for 0.25 seconds; keep torque and off-centre actions out of scope.
- Support 2-4 circles or rectangles with solid randomized colors.
- Store raw RGB as `uint8 [T,H,W,3]`, masks as `bool [T,N,H,W]`, and positions as normalized `float32 [T,N,2]`.
- Store selected masks as `[H,W]` on disk; loaders add the channel and batch dimensions.
- Use contact-onset labels, fixed wall order `left/right/top/bottom`, and replay tolerances of `1e-6` absolute and relative.
- Version the current contracts as `scene-0.1` and `rollout-0.1`.

Evidence: deterministic generation, archive round trips, contact labels, loaders,
and exact replay are covered by the Workstream A test suite and generated pilot.

## D005: ID dataset protocol: 15 October 2026

- Use `dataset-0.1` with 250 visible scene groups and four hidden-physics variants per group, for 1,000 clips.
- Keep frame 0, selected mask, and action identical within each condition while varying mass, friction, and restitution from independent recorded seeds.
- Split scene groups 80/10/10 into train, validation, and test; never split individual variants.
- Store JSON metadata plus compressed NPZ arrays, a dataset manifest hash, and a split manifest hash.
- Treat this 32-frame dataset as the frozen ID integration dataset. Any scale or distribution change creates a new dataset version rather than modifying it in place.

Evidence: all 1,000 samples passed schema, group-integrity, split-leakage, and
exact-replay checks. Collision onset occurs in 8.8% of clips and wall contact in
39.6%.

## Pending decisions

| ID | Decision | Evidence needed | Task |
|---|---|---|---|
| D002 | Exact submission cutoff | NTULearn; PDF lists 16 November, working date remains 14 November | T01 |
| D003 | TC2 training budget and software environment | GPU model/VRAM, remaining quota details, software versions, and benchmark; account access and Slurm limits are recorded in COMPUTE.md | T02-T03 |
| D006 | Autoencoder/backbone dimensions and training budgets | 64px reconstruction / overfit / throughput checks | T08-T10 |
| D007 | Final ablation settings and 128px decision | Validation and compute evidence | T13 |

## D008: Course brief and submission format: 7 October 2026

Use the AI6132 technical-review requirements in [REQUIREMENTS.md](REQUIREMENTS.md). Submit the report PDF, contribution PDF, and source-code ZIP as described in [SUBMISSION.md](SUBMISSION.md). The custom method and demo are project choices, not extra required uploads.

## D009: Pixel pilot recommendation and research audit: 14 October 2026

- Recommend a compact pixel-space space-time U-Net before adding a learned autoencoder.
  This is an engineering proposal for the meeting, not a claim that pixels beat latents.
- A forward/backward-tested prototype implements equal-capacity numeric/spatial
  conditioning and optional trajectory/collision heads. It has no dataset trainer or sampler.
- The original proposal used 16 total frames. The later frozen Workstream A contract
  supersedes it with 32-frame pilot clips and explicit units/force duration in ARCHITECTURE.md.
  The synthetic profiler remains configurable and defaults to 16 frames.
- Force controls, spatial physics maps and object-centric futures already overlap
  Force Prompting, PhyCo, PhysCtrl, and PlaySlot. Define improvement on a frozen matched tabletop protocol rather than
  promise a universal SOTA win. See [research](SOTA_2026-10-14.md) and [build path](BUILD_PLAN.md).
- If the meeting adopts the pixel path or reduces scope, synchronize affected issues,
  task board and backlog dependencies before treating the proposal as the frozen plan.

## D010: TC2 operational preparation: 14 October 2026

- Local preflight and allocation-guarded inventory/bootstrap/probe scripts exist.
- Use assigned QoS and actual driver/OS evidence; guide examples are not allocations.
- Athul's account approval and limits are recorded in COMPUTE.md. First login, network
  connectivity, and an allocated job still need verification. Local CUDA success and
  synthetic timings do not establish the TC2 training budget.
- Keep application details, private connection reports, and restricted PDFs outside Git.

## D011: Pixel path, staffing and finish date: 16 October 2026

- Adopt 64px/32-total-frame pixel diffusion. Remove T08 from T10's critical
  path; autoencoding/latents and 128px remain optional measured follow-ups.
- Core: trained CVAE versus diffusion and matched E2/E3 conditioning. E4 is
  a gated ablation; E1 is diagnostic. Use three seeds for main claims when
  measured budget permits. A failed or unavailable external adaptation is
  reported honestly; it does not turn an internal win into a SOTA claim.
- Preserve contact-only friction and dataset-0.1. PR #28 defines the explicit
  physics description and next-version action/contact coverage for Sanjay.
- Ananya is proposed for CVAE, pending acceptance. Athul/Ananya remain joint
  evaluation owners. Shrijeet owns diffusion, own-account TC2 runs and demo
  integration. See CVAE_HANDOFF.md for the baseline contract.
- Complete presentation, report, contribution PDF and tested source ZIP by
  14 November. The possible 16 November upload
  cutoff does not extend that commitment.
- PR #25 and #26 passed independent review and merged. Their tests are
  implementation/control evidence; the new trainer must still demonstrate
  coherent generated video and held-out action response before scale-up.

## D012: CVAE ownership confirmed: 17 October 2026

- Ananya accepted the CVAE lead. Athul and Ananya remain joint evaluation owners.
- Athul authored the merged CVAE implementation and training diagnostics. Ananya
  takes subsequent CVAE training, prior-quality checks and validation; ownership
  does not establish past contributions.
- Shrijeet keeps diffusion, own-account TC2 runs and demo integration. Sanjay keeps
  simulator/data. All deliverables remain due internally on 14 November.

## D013: E0 baseline, action horizons and force-OOD: 24 October 2026

- E0 is the plain flow-decoder CVAE (`configs/training/cvae_flow_pilot.yaml`). On
  validation it tracks 5,654/5,952 observations with 80/96 complete clips. The RGB
  CVAE tracks 398 with none, so most physical metrics are undefined for it. The RGB
  CVAE stays in the report as a decoder ablation. The displacement-supervised flow
  variant uses future positions in its loss and stays a separate follow-up.
- E0 uses training seeds 0-2 under the frozen protocol. The flow decoder warps
  frame-0 pixels, so the CVAE-diffusion comparison also differs in output
  representation; the report states this.
- Action controls gain a supplementary full-horizon score
  (`evaluation/action_horizon.py`, version 1). It reruns the unchanged action,
  reversal and selected-object scorers at the final frame, only where every
  hidden-physics reference is contact-free until then. The original 0.5-second
  scores stay unchanged and reported. At native force 11/16 continuous and 8/16
  rendered reference pairs fall below the threshold inside that window.
- Force-OOD is evaluated for E0/E2/E3 as labelled magnitude extrapolation and
  reported per action family. E5/E6 declare a 1.1 force limit and are reported as
  unsupported unless they enter the final comparison under a reviewed contract.
  Forward, reverse and reversal scores run on force-OOD; only zero force is absent.
- These choices follow validation results, including the flow-CVAE timing
  diagnostic in #52. No test split was opened. Both horizons are reported for
  every method; neither is chosen per model.
- Proposed by Ananya in #52 and her review of #50; confirmed by Athul. Final
  checkpoints, diffusion recipes and the training budget remain open under T13.

## D014: Final E2/E3 runs and opening the test splits: 25 October 2026

- E2/E3 use the 17 October pilot recipes with training seeds 0-2
  (`configs/training/diffusion_repeats/`). Each config is the pilot's effective
  config with only the seed changed: 5,000 updates and the final EMA checkpoint.
  The diffusion training code is unchanged since the pilot. The seed-7 pilot
  checkpoints stay as pilot records and are not part of the final comparison.
- TC2 job 46704 trained the six runs and scored them with the three E0 seeds on
  validation. E2 tracks 176, 140 and 1 of 5,952 observations and E3 5, 546 and 2,
  with no complete clip in any run. The pilot's seed-7 runs tracked 378 and 269.
  The failure is consistent across seeds. See results/diffusion-seeds-validation.json.
- Diffusion sampling uses 32 DDIM steps, the pilot default. The 16/32/64-step study
  had no complete clip at any setting and selected no sampler, so the declared
  default is kept rather than chosen on tracking counts.
- Everything else stays as declared in `configs/evaluation/comparison.yaml` and
  EXPERIMENT_PROTOCOL.md, and force-OOD runs without the zero-force action plan:
  - sampling seeds 0-3, tracker version 2 and the pixel config;
  - SSIM/LPIPS and both action horizons;
  - the three collision views, the missing-track policy and the scene bootstrap;
  - the force-0.4 selection probe.
- The five reserved splits are opened although no method passes the validation
  gate in `configs/training/multiscene_pilot.json`. The report needs held-out
  results, and they are reported as measured failures. No checkpoint, setting or
  method changes after test results are seen.
- One account (Athul's TC2) prepares the plans and samples and scores all nine
  runs with `cluster/final_evaluation.sbatch` in one job capped at two hours.
  This is required because sampled predictions are bound to plan hashes, plans
  store absolute paths and TC2 home folders are private. Ananya analyses E0;
  Athul analyses E2/E3.
- Agreed by Ananya and Athul on 25 October, following Ananya's T13/T14 proposal.

## D015: Budget-matched E2/E3 as a secondary test result: 25 October 2026

- This decision was made after the D014 test results were seen. Its results are
  reported as a secondary, post-test comparison, labelled as such. The D014
  results stay the primary comparison and are not replaced.
- Reason: the D014 E2/E3 runs saw 5,000 training clips and E0 saw 40,000. On
  validation (job 46733), training E2/E3 seed 0 for 40,000 updates removes the
  noise and raises SSIM from 0.45 to 0.93, but only 5-7 of 96 clips are fully
  tracked. See results/diffusion-long-validation.json.
- E2/E3 use the D014 configs with only `steps` set to 40,000
  (`configs/training/diffusion_long/`), training seeds 0-2. The cosine schedule
  runs over 40,000 updates.
- The 40,000-update checkpoint is used for every seed. It is fixed here as the
  budget-matched endpoint and is not chosen on validation or test scores. The
  10k and 20k checkpoints are validation records only.
- Everything else is as in D014: 32 DDIM steps, the same splits, plans, scorer,
  sampling seeds and force-OOD handling, from one account with
  `cluster/final_evaluation.sbatch` and `C2F_DECISION=D015`.
- The three E0 seeds are sampled again in the same job. They must reproduce the
  D014 numbers, which checks that the evaluation is repeatable. The validation
  and test jobs are held to the A40 nodes (TC2N01-02) used for D014, because
  another GPU model may change results in the last digits. Training seeds 1-2
  ran on an L40S node (TC2N03).
- The nine runs are scored on validation first with
  `cluster/final_rehearsal.sbatch`. That run is a record, not a gate.
- No further training, checkpoint or setting changes follow from these results.
- Proposed by Athul on 25 October after the D014 results, for team review on #64.
- Wording note added after review on #64: "budget-matched" here means the same
  number of training clips seen (40,000). It does not match updates (40,000 at
  batch 1 against E0's 20,000 at batch 2) or GPU time.

## D016: Numeric versus spatial force routing in the pose model (E10): 2 November 2026

- Reason: the E2-versus-E3 question cannot be answered in pixel space, because
  neither model keeps object identity (D014, D015). The pose model keeps objects
  intact, so the same question can be measured there.
- E10 has two arms on the E6 canvas-loss recipe (`contact_wall_pilot.json`):
  - spatial: the force reaches only the clicked object's token, as in E5/E6;
  - numeric: both object tokens get the same force. The selected-object flag stays
    in the features, so the model must learn which object to move.
- Both arms have the same 845,828 parameters. For each training seed they start
  from identical weights and draw identical batches, noise levels and noise. A
  test checks that the spatial arm trains exactly like E6.
- The only other changes from E6, in both arms, do not change training updates:
  - `force_limit` 1.5 instead of 1.1. It is an input check (training forces stay
    at most 1.1) and allows force-OOD (1.3/1.5) to be scored as extrapolation;
  - `max_seconds` 2400 instead of 1200. On an A40 the 16,000 updates take about
    1,210 seconds, so the old stop would end runs early. Job 50249 was cancelled
    for this reason during the first training run, before any evaluation.
- Named E10 because #69 registers its surface model as E9.
- Training seeds 19, 20 and 21; 16,000 updates; `best.pt` is chosen by the unchanged
  fixed-seed validation denoising loss. Nothing else is chosen on scores.
- Validation uses the existing 24 conditions, the force-0.4 selection probe, the
  passive check, the geometry audit and the wall-clipping check, for every run.
- Main measures, fixed now: correct-target selection pairs (0.5 s window and full
  clip), unselected-object drift, early reversal, complete clips and penetration.
  A difference is called consistent only if it has the same sign in all three
  seeds. Every run and failure is reported.
- Held-out results come only from the separate D017 corpus. The dataset-0.3
  reserved splits are not used for E10.
- Scope: E10 uses future poses in training. It answers the routing question for a
  pose-space generator, not for E2/E3 pixel diffusion.
- Proposed by Athul on 2 November, after a review that found the central comparison
  unresolved. Run with `submission/routing_comparison.sbatch`.
- Results (job 50264, `docs/results/routing-validation.json`): all six runs completed
  16,000 updates. On validation both arms select the right object in every pair with
  no drift. Penetrating clips (over 1 px): spatial 14/11/11, numeric 17/16/14 for
  seeds 19/20/21, so numeric is worse in all three seeds. Complete clips are mixed
  (spatial 94/96/96, numeric 96/92/95).

## D017: Held-out replication on a separate, larger corpus: 2 November 2026

- Reason: each dataset-0.3 reserved split has two scene groups, so the scene
  bootstrap intervals are not informative.
- A new corpus from the same generator, seed 790000, with ten scene groups per
  reserved split (50 groups instead of 10). Its one training and one validation
  group are generated by the generator but not used.
- Every plan records the checkpoints' training corpus (dataset-0.3) and refuses to
  build if any evaluated clip shares a frame zero or visible seed with any
  dataset-0.3 clip.
- Frozen runs, with no training, selection or tuning: E0 seeds 0-2 (D014), E2/E3
  5k seeds 0-2 (D014), E2/E3 40k seeds 0-2 (D015) and the six E10 runs (D016).
- Same scorer, sampling seeds 0-3, samplers (32 DDIM steps; 16 Heun steps for E10),
  force-0.4 selection probe and force-OOD handling as D014.
- Two jobs on the A40 nodes: E0 with E2/E3 40k, then E2/E3 5k with E10. Both
  prepare the same plans; their plan hashes must match.
- D014 and D015 stay as recorded. These results are reported as a replication
  beside them, labelled as such, whatever they show.
- Evaluator change: a plan can now bind a separate training corpus, and the pose
  sampler may open test splits only on such a corpus. Earlier plans replay at
  their own source revisions, as already documented for D014/D015.
- Proposed by Athul on 2 November. Run with `cluster/replication_evaluation.sbatch`.
- Results (2-3 November; scores in `docs/results/replication-d017.json`, from
  `scripts/analyse_replication.py`). The run took three jobs: 50265 (E0, E2/E3 40k),
  50321 (E2/E3 5k and E10 on the first three splits; a disk quota error stopped it
  in force_ood) and 50527 (the same for force_ood and physics_test, with output kept
  on the node and compressed afterwards). Main plan hashes match across jobs. The
  selection plan hashes differ only because each embeds its output path.
  - Pixel models: E2 and E3 at 5k lose almost every object track (missing rate 0.998
    to 1.0 on every split). At 40k E2 selects the right object in 0 to 8% of pairs
    and E3 in 0 to 50%; E0 selects 0%. This repeats the D014/D015 finding on 50 new
    scene groups.
  - E10 (pose model, 3 seeds per arm): both arms select the right object in 86 to
    100% of pairs on every split with no drift. Missing-track rates are at most 0.002
    on id_test, texture_ood and physics_test, 0.017-0.020 on shape_ood and 0.04-0.25
    on force_ood.
  - Spatial versus numeric: spatial has the lower selected-object ADE in all three
    seeds on id_test, shape_ood, texture_ood and physics_test (for example id_test
    0.026-0.027 versus 0.028-0.030), so numeric routing is not better there. ADE
    averages complete trajectories within each case, so a defined value does not
    establish complete tracking of every clip. It rests on 10 of 10 scene groups in id_test,
    texture_ood and physics_test and 9 of 10 in shape_ood. In force_ood only 0-5 of
    10 groups are scored (numeric 3/1/3, spatial 1/5/0 for seeds 19/20/21), so its
    ADE is not compared. The track-loss rate covers all groups: numeric 0.04-0.13,
    spatial 0.12-0.25. Numeric loses fewer observations in seeds 19/21; seed 20
    reverses that order (numeric 0.1268, spatial 0.1219). These are descriptive
    per-seed rates, not a consistent advantage across every seed.
    Selection is not consistent: shape_ood has spatial 0.94/0.86/0.86 and numeric
    0.86/0.92/0.86. Validation (D016) already showed more penetration for numeric.
  - All six E10 runs train from fresh initial weights (no E7 teacher), one per seed;
    the two arms of a seed share weights and batches. The seeds are independent
    initializations, not refinements of one model.
  - Scope: one corpus, 10 scene groups per split, 3 seeds. The pixel and pose models
    differ in more than routing, so E10 does not answer the E2-versus-E3 question
    for pixel diffusion.

## D019: Expected direct-decoder loss investigation

- Keep D014-D018 and all negative results unchanged.
- Diagnose the unnecessary random EDM weighting of deterministic E11 predictions.
- Compare original and analytically expected direct losses using the same E7
  teacher, frozen cache, seed, batches and 8,000-update budget. The expected
  coefficients include both clipped distribution tails and the original
  geometry activation probability. This changes gradient variance, not inference.
- Use original validation only. A result on the opened D017/D018 sets cannot
  support a new confirmatory claim or select this candidate.
- Preserve the original physical gates and report every regression. A successful
  software test or lower training loss cannot promote a physical model.
- Full checkpoints, detached Slurm submission receipts and resumable checked
  transfers protect the run against intermittent VPN access.
- See [the investigation](PHYSICAL_QUALITY_REPAIR.md). Training outcome is pending.

## D020: Rest-anchored physical-response repair

- D019 still has two drifting zero-force validation samples out of sixteen.
- Train and sample a shared neural force response minus its zero-force response.
  This guarantees rest without a simulator or a test-only output override.
- One seed, the same E7 teacher/cache and 8,000 final updates; expected loss,
  half-pixel training margins and fourfold penetration/canvas loss weights.
- This is a combined candidate, not a single-component causal comparison.
  Keep the scorer, sample seeds, quality gates and historical results unchanged.
- Record two backbone evaluations per direct video. Validate every action and
  contact metric; fresh-scene generalization remains unverified.

## New decision template

```text
ID / date:
Decision:
Alternatives considered:
Evidence / linked issue or run:
Tradeoffs:
Effect on interfaces, experiments, and deadline:
Owner / reviewers:
```

## D021: Research-backed observable-physics baseline and fresh ID comparison

- Review primary methods, released code and compute before further model changes;
  see PHYSICS_METHOD_REVIEW_2026-11-03.md. No general video leaderboard proves a
  winner on our two-object task.
- Adapt the explicit physical-state/observed-sprite mechanism supported by PhysGen
  and recent IMPACT, with sampled hidden properties. Label it physics-assisted,
  not a neural model, full paper reproduction or exact simulator reference.
- Analytical checks cover rest, force impulse, mass scaling, contact momentum,
  dissipative energy and integration convergence. Original validation retains
  the existing RGB tracker and one-pixel geometry threshold.
- Before new data generation, freeze the physics implementation and D020 final
  8,000-update checkpoint. Generate seed 910000 with ten ID-test scene groups and
  one group in each other required split. Score ID only, all 60 conditions and
  four seeds 0-3 for each generator. Required remaining splits stay unscored.
- Bind the original corpus and reject shared initial images/visible seeds. Keep
  every condition, missing track, contact error and failure. No tuning, filtering
  or checkpoint selection on this fresh corpus. Historical gates remain intact.

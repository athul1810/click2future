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

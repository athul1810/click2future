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

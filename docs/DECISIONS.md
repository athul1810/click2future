# Decision log

## D001: Initial technical scope: 7 October 2026

- Compare a conditional VAE and video diffusion for action-conditioned tabletop futures.
- Test spatial action conditioning and auxiliary trajectory/collision supervision separately.
- Start at 64px; consider 128px after profiling on TC2.

## Pending decisions

| ID | Decision | Evidence needed | Task |
|---|---|---|---|
| D002 | Group details and exact submission cutoff | Confirmed roster and NTULearn; PDF lists 16 November, working date remains 14 November | T01 |
| D003 | TC2 training budget and software environment | Application/access status, hardware details, quota, and benchmark | T02-T03 |
| D004 | Simulator engine, units, friction model, sample schema | Small deterministic prototype | T04-T05 |
| D005 | Dataset size, split definitions, hidden-property distributions | Pilot coverage and storage profile | T06 |
| D006 | Autoencoder/backbone dimensions and training budgets | 64px reconstruction / overfit / throughput checks | T08-T10 |
| D007 | Final ablation settings and 128px decision | Validation and compute evidence | T13 |

## D008: Course brief and submission format: 7 October 2026

Use the AI6132 technical-review requirements in [REQUIREMENTS.md](REQUIREMENTS.md). Submit the report PDF, contribution PDF, and source-code ZIP as described in [SUBMISSION.md](SUBMISSION.md). The custom method and demo are project choices, not extra required uploads.

## D009: Pixel pilot recommendation and research audit: 14 October 2026

- Recommend a compact pixel-space space-time U-Net before adding a learned autoencoder.
  This is an engineering proposal for the meeting, not a claim that pixels beat latents.
- A forward/backward-tested prototype implements equal-capacity numeric/spatial
  conditioning and optional trajectory/collision heads. It has no dataset trainer or sampler.
- Preserve the proposed 16-total-frame, centre-force contract until the generator's
  raw fixture/metadata is handed off. Units and force duration remain unresolved.
- Force controls, spatial physics maps and object-centric futures already overlap
  Force Prompting, PhyCo, PhysCtrl, and PlaySlot. Define improvement on a frozen matched tabletop protocol rather than
  promise a universal SOTA win. See [research](SOTA_2026-10-14.md) and [build path](BUILD_PLAN.md).
- If the meeting adopts the pixel path or reduces scope, synchronize affected issues,
  task board and backlog dependencies before treating the proposal as the frozen plan.

## D010: TC2 operational preparation: 14 October 2026

- Local preflight and allocation-guarded inventory/bootstrap/probe scripts exist.
- Use assigned QoS and actual driver/OS evidence; guide examples are not allocations.
- Account/network approval and an actual allocated job are still required. Local
  CUDA success and synthetic timings do not certify TC2 or establish the training budget.
- Keep application details, private connection reports, and restricted PDFs outside Git.

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

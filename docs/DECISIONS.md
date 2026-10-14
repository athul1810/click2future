# Decision log

## D001: Initial technical scope: 7 October 2026

- Compare a conditional VAE and video diffusion for action-conditioned tabletop futures.
- Test spatial action conditioning and auxiliary trajectory/collision supervision separately.
- Start at 64px; consider 128px after profiling on TC2.

## Pending decisions

| ID | Decision | Evidence needed | Task |
|---|---|---|---|
| D002 | Exact submission cutoff | NTULearn; PDF lists 16 November, working date remains 14 November | T01 |
| D003 | TC2 training budget and software environment | GPU model/VRAM, remaining quota details, software versions, and benchmark; account access and Slurm limits are recorded in COMPUTE.md | T02-T03 |
| D004 | Simulator engine, units, friction model, sample schema | Small deterministic prototype | T04-T05 |
| D005 | Dataset size, split definitions, hidden-property distributions | Pilot coverage and storage profile | T06 |
| D006 | Autoencoder/backbone dimensions and training budgets | 64px reconstruction / overfit / throughput checks | T08-T10 |
| D007 | Final ablation settings and 128px decision | Validation and compute evidence | T13 |

## D008: Course brief and submission format: 7 October 2026

Use the AI6132 technical-review requirements in [REQUIREMENTS.md](REQUIREMENTS.md). Submit the report PDF, contribution PDF, and source-code ZIP as described in [SUBMISSION.md](SUBMISSION.md). The custom method and demo are project choices, not extra required uploads.

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

# Decision log

## D001: Project and repository scope: 7 October 2026

- Decision: use Click2Future as the working GenAI project and create a separate private repository.
- Basis: the supplied project brief, instructor feedback, and team request.
- Repository: `athul1810/click2future`; BudgetGate is a workflow reference only.
- Initial state: planning documents and empty implementation folders; no trained models yet.
- Ownership: all task owners unassigned.
- Compute: use the course-provided TC2 GPU Cluster. Submit the course application form to CCDS GPU Support and record access status, hardware, and usage limits in T02.
- Scope: start at 64px; gate 128px on measured feasibility.

## Pending decisions

| ID | Decision | Evidence needed | Task |
|---|---|---|---|
| D002 | Exact course deliverables and cutoff | Course site / rubric | T01 |
| D003 | TC2 training budget and software environment | Application/access status, hardware details, quota, and benchmark | T02-T03 |
| D004 | Simulator engine, units, friction model, sample schema | Small deterministic prototype | T04-T05 |
| D005 | Dataset size, split definitions, hidden-property distributions | Pilot coverage and storage profile | T06 |
| D006 | Autoencoder/backbone dimensions and training budgets | 64px reconstruction / overfit / throughput checks | T08-T10 |
| D007 | Final ablation settings and 128px decision | Validation and compute evidence | T13 |

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

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
| D002 | Group details and exact submission cutoff | Confirmed roster and NTULearn; PDF lists 16 November, working date remains 14 November | T01 |
| D003 | TC2 training budget and software environment | Application/access status, hardware details, quota, and benchmark | T02-T03 |
| D004 | Simulator engine, units, friction model, sample schema | Small deterministic prototype | T04-T05 |
| D005 | Dataset size, split definitions, hidden-property distributions | Pilot coverage and storage profile | T06 |
| D006 | Autoencoder/backbone dimensions and training budgets | 64px reconstruction / overfit / throughput checks | T08-T10 |
| D007 | Final ablation settings and 128px decision | Validation and compute evidence | T13 |

## D008: Course brief and submission format: 7 October 2026

- Source: the supplied `course project technical review.pdf` and accompanying submission instructions.
- Course: AI6132 Generative AI For Visual Synthesis, Project 1: Technical Review.
- Minimum: two methods, one evaluation dataset, and a report of at least four pages with the required analysis.
- Submit one group package consisting of `group_XX_report.pdf`, `group_XX_contribution.pdf`, and `group_XX_code.zip`.
- Exclude datasets, model files, external links, and cloud-hosted substitutes from the submitted artifacts.
- Keep 14 November as the working deadline at the user's request until the discrepancy with the PDF's 16 November date is confirmed.
- Group ID and coordinator remain to be recorded. Slides and a separate demo video are not required by the supplied instructions.

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

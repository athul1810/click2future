# Roadmap

**Submission deadline: 14 November 2026.** Confirm the exact cutoff on the course site.

Intermediate milestones and task dates have not been set. The team will agree on them later. The dependencies below describe the order of work, without assigning dates.

## Critical path

```mermaid
flowchart LR
    T04[Data contract] --> T05[Simulator]
    T05 --> T06[Dataset and splits]
    T06 --> T08[Autoencoder]
    T08 --> T10[Diffusion baselines]
    T10 --> T11[Spatial conditioning]
    T11 --> T12[Auxiliary heads]
    T12 --> T14[Core ablations]
    T14 --> T19[Reproduction and release]
    T19 --> T20[Submission]
```

Metrics, the CVAE, report outline, and demo shell can progress alongside the critical path using agreed fixtures. Integration still requires the real dataset and checkpoints.

## Checkpoints

**Compute and setup.** Record actual GPU access and a first memory/throughput estimate. If allocation is pending, keep CPU simulator and contract work moving and identify a compute fallback without assuming access.

**Check the 64px baseline.** Require a small-batch overfit check, interpretable reconstructions, valid samples, and measured training cost. Debug data/model issues before launching the ablation suite.

**Fix the experiment settings.** Fix datasets, metrics, sample counts, and run budgets. Scale to 128px only if the 64px evidence and remaining GPU budget justify it. Use 128px for a targeted follow-up, not an automatic second full sweep.

**Prepare for submission.** Complete the analysis, reproduction checks, report, and presentation. Maintain a recorded demo if live sampling is slow or cluster access is unavailable.

## Team coordination

- Keep each task's GitHub issue updated with its owner, branch, blockers, and evidence.
- Review interfaces before merging data/model/evaluation changes.
- Test a full scene-to-samples-to-metrics path when components are ready to integrate.
- Link runs in [experiments/](../experiments/) and measured tables in [results/](../results/).
- Raise blocked tasks at the next team check-in and reassign work if needed.

## Reduced-compute fallback

Keep 64px as the final resolution, use shorter clips if profiling requires it, cap dataset size, and start with one seed for smoke runs. Prioritize the E2→E3→E4 contribution comparisons while preserving E0/E1 as small-budget contextual baselines. Use additional seeds for the main comparisons when affordable, disclose seed counts, and avoid overstating single-run differences. FVD and friction-conditioned generation can be dropped without changing the main question.

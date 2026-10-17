## Change

What problem does this solve, and what can a teammate now do?

## Issues

Use `Closes #123` for each completed issue and `Part of #123` when work remains.
Repeat the keyword for every issue: `Closes #123` and `Closes #124`, not `Closes #123, #124`.
Write `No linked issue` only for maintenance outside the task board.
GitHub closes linked issues when the changes merge into `main`. The linked-issues
check fails if this section names no issue.
For stacked PRs, merge the parent first and retarget this PR to `main`.

## Validation

Commands, checks, run IDs, or visual evidence:

## Interface / experiment impact

Schema changes, dataset versions, affected comparisons, and limitations:

## Checklist

- [ ] Relevant documentation updated.
- [ ] No credentials, full datasets, or checkpoints committed.
- [ ] Measured results are traceable to configs and seeds, if applicable.

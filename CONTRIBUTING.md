# Contributing

## Claim and scope work

1. Open the [task board](docs/TASKS.md) and choose an unassigned issue whose dependencies are ready.
2. Assign one lead and agree on a reviewer. Workstream labels describe the work, not a person's fixed role.
3. Add a short implementation plan and identify any interface changes before coding.
4. Create a branch such as `feat/t05-simulator` or `docs/t01-course-requirements`.
5. Open a pull request linking the task (`Closes #...`) with the relevant evidence.

When changing a task's scope or dependencies, update its GitHub issue, [task board](docs/TASKS.md), and [backlog](docs/task_backlog.json) together. Keep live ownership and progress in GitHub Issues.

## Definition of done

- The issue's acceptance criteria are met and its dependencies are accounted for.
- A teammate can follow the documented setup/command and reproduce the relevant output.
- Meaningful checks cover changed physics, tensor contracts, metrics, or model behavior.
- Config, seeds, run/checkpoint references, and limitations are recorded where relevant.
- Interface changes are reflected in [ARCHITECTURE.md](docs/ARCHITECTURE.md).
- A reviewer has checked the change before merging to `main`.

## Experiment discipline

Use the [run manifest template](experiments/RUN_TEMPLATE.md). Never compare unnamed checkpoints or tune on test sets. Share fixed fixtures and small smoke runs before scheduling expensive experiments. Keep simulator output and model-generated output visibly labelled.

## Files and artifacts

Commit code, docs, configuration, split manifests, small tables, and selected report figures. Keep full datasets, model checkpoints, large videos, logs, and credentials outside Git. Store artifact paths and checksums in run manifests; document how teammates obtain them. The `.gitignore` contains initial exclusions.

Those artifact references are for team work. The course submission must contain the two PDFs and source-code ZIP described in [SUBMISSION.md](docs/SUBMISSION.md), with no datasets, model files, or external links. Prepare a separate code package rather than ZIP the whole working repository.

## Current setup status

Follow [development setup](docs/SETUP.md) for the dependency lock, environment check,
and development checks. T03's TC2 validation remains pending access and hardware details
from T02. An untrained denoiser prototype and synthetic profiler are available; dataset
training, sampling, simulator, evaluation, and demo implementations are not available yet.
See [the build plan](docs/BUILD_PLAN.md) for the prototype checks and remaining evidence gates.

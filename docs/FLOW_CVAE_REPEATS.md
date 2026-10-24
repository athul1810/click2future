# Flow CVAE training-seed repeats

Pending team agreement (T14): if the unsupervised flow CVAE represents E0 in the
final comparison, repeat its training with independent seeds before main runs.
Seed 0 is the existing run, TC2 job 46333 (`cvae_flow_pilot.yaml`, checkpoint
`df3c5515…`, recorded in `docs/results/cvae-study-checks.json`). The configs in
`configs/training/cvae_repeats/` change only `training.seed` to 1 and 2. The seed
sets initial weights, data order and training noise; data, model, objective,
20,000-update budget and checkpoint rule are unchanged.

## Run

Upload the committed source with `scripts/tc2_setup.sh -s Upload -r <commit>`, then
on TC2 reuse the validated environment and the prepared pilot corpus:

```sh
export C2F_PROJECT=$HOME/click2future/releases/<commit>
export C2F_PYTHON=$HOME/click2future/releases/<environment-release>/.venv/bin/python
export C2F_SOURCE_REVISION=<commit>
export C2F_PILOT_ROOT=$HOME/click2future/pilot-0.3
export C2F_STUDY_DIR=$HOME/click2future/cvae-study
cd "$C2F_PROJECT"
sbatch --time=01:00:00 cluster/cvae_study.sbatch configs/training/cvae_repeats/flow_seed1.yaml configs/training/cvae_repeats/flow_seed2.yaml
```

Each seed trains in about 12 minutes on an A40 and is then evaluated and probed,
so the job needs about 30 minutes. Copy results back without checkpoints, then
back up `best.pt` separately and verify its hash against `eval/cvae-results.json`.

## Analyse

On CPU, for each seed run, with the local corpus copy:

```sh
python scripts/summarize_cvae_study.py outputs/tc2-ops/cvae-study
python submission/cvae_motion_timing.py --plan <run>/eval/plan.json --data outputs/tc2-ops/pilot-0.3/data --samples <run>/eval/validation-samples --output <run>/motion-timing-validation.json
python submission/cvae_motion_timing.py --plan <run>/eval/target-controls/plan.json --data <run>/eval/target-controls/references --samples <run>/eval/selected-object-samples --output <run>/motion-timing-selection.json
python submission/cvae_selection_drift.py --evaluation <run>/eval --dataset outputs/tc2-ops/pilot-0.3/data --output <run>/selection-drift.json
```

Report all three seeds, including failures, with the mean and range across seeds
for tracking, complete clips, full-horizon displacement and selection drift. The
four sampling seeds per condition do not replace these training seeds. No test
split is opened.

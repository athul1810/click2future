# Multi-scene CVAE and physics pilot

This is a bounded validation experiment for T12-T16. It is separate from the
one-clip dataset-0.1 checks and is not the final method comparison. The training
source is `2ac7710953c6754ab3404ec104b50234386e19b1`; TC2 training job 43291 uses
the fixed settings in `configs/training/multiscene_pilot.json`.

## Fixed inputs and budgets

Dataset-0.3 seed 730000 gives 16 training scene groups, four validation groups and
two groups in each of the five test splits. There are 336 training clips, 84
validation clips and 248 reserved test clips, all at 64px, 32 frames and 8 FPS.
All 668 clips passed exact replay before training. Scene groups and their actions
and physics variants stay together. This seed range does not overlap the reserved
extractor controls.

The CVAE uses width 16, latent dimension 64, seed 0, batch size 2 and at most
20,000 updates. Full validation posterior loss plus KL selects `best.pt` every
1,000 updates. The per-process training limit is 30 minutes. Prior samples,
not posterior reconstructions, enter the generated-video evaluation.

Numeric E2, spatial E3 and auxiliary E4 use independently initialized width-32
velocity diffusion, RGB targets, seed 7, batch size 1 and 5,000 updates each.
They share the same training clips, EMA decay, noise schedule, learning rate,
foreground weight 32 and two-pixel foreground margin. E4 adds trajectory/contact
heads and their losses. Its extra parameter initialization also consumes RNG;
the common seed policy does not assert identical E3/E4 noise draws.

The contact positive weight is computed once from training labels only:
156 positive and 10,260 negative unordered pair intervals give a capped weight
of 20. The effective configs and their hashes are saved with the run. Future
masks weight training errors only; they never enter generator conditioning.

The pilot permits at most 1.5 allocated GPU-hours including evaluation. This is
a cap, not observed cost or an account quota. Different CVAE/diffusion update and
batch budgets, one training seed and four validation groups prevent a final
method ranking or training-seed uncertainty claim. SSIM is the declared pilot
image metric; the final comparison still needs LPIPS and its remaining studies.

## Reproduce

From the configured source directory, CPU preparation is allowed:

```sh
python -m click2future.training.multiscene_pilot --root outputs/new-pilot --prepare-only
```

The trainer refuses execution outside an allocated TC2 GPU job. Set the committed
source, validated interpreter, a new pilot directory and source revision, then:

```sh
sbatch submission/train_pilot.sbatch
```

The script reads `C2F_PROJECT`, `C2F_PYTHON`, `C2F_PILOT_ROOT` and
`C2F_SOURCE_REVISION`. To use separately prepared data in an allocation, run the
same module with `--train-only`. Frozen input/config hashes must still match.
Failed or existing runs are retained; choose a new directory for another attempt.

After training completes, set `C2F_PILOT_RESULTS` to a new directory and submit:

```sh
sbatch submission/evaluate_pilot.sbatch
```

Use the original training runner source bytes so the frozen-pilot check remains
valid. The evaluation source may add analyses while keeping the trained model
implementation unchanged. Record its own revision separately.

## Evaluation

- Every validation condition receives four prior/EMA samples with seeds 0-3.
  All reference physics variants enter trajectory coverage and spread reports.
- Zero/reversed-force checks use the frozen 0.5-second action window. Simulator
  labels and rendered references supply separate calibration; weak motion and
  missing tracks stay visible.
- Selected-object checks hold image, force, duration and sample seed fixed.
  The shared force magnitude is 0.4, chosen in the earlier validation control.
  Derived references are separate from the training corpus.
- E4 heads come from actual timestep-zero DDIM calls. Replaying with capture must
  reproduce every saved RGB pixel. Head errors, RGB errors and their direct
  agreement have separate reports. Missing RGB observations cannot count as
  agreement. Collision extraction from RGB remains approximate.
- CVAE diagnostics cover every training and validation clip, reporting posterior
  reconstruction, prior error, per-latent KL and prior RGB spread. Low KL or small
  pixel spread alone does not prove posterior collapse or meaningful uncertainty.

Extractor `palette-components-2` is the candidate under review in PR #42.
These scores must not be mixed with version-1 results. The strict trajectory
engineering gate is separate from multi-reference uncertainty and action checks;
passing it does not prove physical validity. Test splits remain reserved until
validation quality and action response support proceeding. No threshold is tuned
on generated test outputs.

The measured result record will distinguish completed computation from failed
quality. A completed Slurm job alone does not finish the research tasks.

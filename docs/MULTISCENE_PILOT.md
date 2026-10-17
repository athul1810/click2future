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
method ranking or training-seed uncertainty claim. SSIM is the declared initial pilot
image metric; supplemental CPU LPIPS on the frozen videos is now measured.

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

## Reserved sampler study

The earlier protocol reserved 16/32/64 DDIM steps on one fixed checkpoint and at
most 16 validation conditions, capped at 0.5 allocated GPU-hours. The pilot uses
E3 and the first four sorted validation scenes, with zero, forward, reverse and
collision-0-to-1 actions in each. This subset is fixed before generated outputs
are read; all omitted cases remain listed. It is separate from the full 24-case
validation report. Checkpoint, sample seeds and metric settings stay unchanged.

After validation, set `C2F_STUDY_RESULTS` and submit
`submission/study_sampler.sbatch`. Its parent is the full validation plan in
`C2F_PILOT_RESULTS`. The study also stays within the combined 1.5-hour pilot cap.
It reports quality, action response and sampling time at every setting, including
failures. More sampling steps alone do not establish useful video quality.

## CVAE flow-decoder follow-up

The completed RGB CVAE still changes object colours and shapes on unseen forward
scenes. A CPU inspection of four fixed validation conditions tracked only four
of 992 future object observations. Those samples are an additional inspection,
not the primary GPU comparison. The RGB run and its checkpoints are retained.

The separate `cvae_flow_pilot.yaml` changes the CVAE decoder to predict two
backward-displacement channels. It warps observed frame-0 pixels with bilinear
border sampling. The output layer starts at zero, so initialization copies the
image. The encoder, posterior/prior, latent dimension, data, seed, objective and
20,000-update budget stay the same. The flow model has 574,226 parameters, versus
574,659 for RGB. This is a decoder and initialization follow-up, not a pure
latent-variable ablation or a claim of novelty.

Flow is bounded to half the image width/height. It can translate or deform observed
appearance but cannot invent textures or recover genuinely unseen content. It
does not enforce rigid motion or collision laws. Static copying, distorted shapes
and incorrect action response must still fail the same evaluation checks.

Sampling uses the same learned conditional prior and observable inputs. Future
RGB remains posterior/loss supervision only. The explicit bilinear implementation
has deterministic gradients to flow without a GPU atomic backward to observed
image pixels. Its values and flow gradients match the standard border sampler
in CPU regression checks. Legacy RGB checkpoint construction is unchanged, and
16 saved CPU prior samples replay pixel-exactly through the new factory.

Train this variant in a separate output folder on the frozen pilot data:

```sh
sbatch --time=00:25:00 submission/train_cvae.sbatch --config configs/training/cvae_flow_pilot.yaml
```

Set `C2F_PROJECT`, `C2F_PYTHON`, `C2F_DATA_DIR`, `C2F_RUN_DIR` and
`C2F_SOURCE_REVISION`. Keep the original pilot and run directories intact.
In an allocation, `submission/flow_evaluation.py` accepts `--root`, `--checkpoint`,
`--output` and `--device cuda`. It evaluates all validation actions, both selected
objects and posterior/prior diagnostics using a fresh source-bound plan. This
follow-up also stays within the combined pilot budget; it does not replace the
original E0 results.

## Motion supervision and E4 collision check

The unsupervised flow decoder preserved appearance but failed the short-window
force and selection controls. The separate `cvae_flow_motion_pilot.yaml` adds a
weight-1 displacement loss. It compares the mask-average backward flow, scaled
to normalized world units, with simulator object-centre displacement. On the
training circles/rectangles the rotation term averages to zero up to rasterization.
Future masks and positions enter only this training loss, never generator inputs.
The decoder, initialization, latent distribution and RGB/KL objective are unchanged.
This is additional supervision and must not be described as an ordinary unsupervised
CVAE baseline or a decoder-only improvement.

Validation checkpoint selection includes the same displacement term. The helper
logs its raw loss and displacement MAE separately. A low field loss still does not
prove correct RGB motion; force/selection controls and pixel tracking remain the
acceptance checks. The new run uses 20,000 updates with an 18-minute process cap
and the same frozen training/validation corpus. Its original unsupervised results
are retained.

The remaining E4 overfit check uses the first sorted training sample,
`g03-scene-00730000-collision_0_to_1-p00`. Its simulator RGB tracks all 62 future
observations and contains one object-contact onset. The fixed 5,000-update config
sets `limit_train=1`; one positive and 30 negative intervals give capped weight 20.
Submit `submission/check_e4_collision.sbatch` with `C2F_E4_CHECK` pointing to a new
directory. It trains, samples four futures and checks actual final DDIM heads,
RGB tracking and collision scores. This is a training-only mechanics check, not
generalization or conditional uncertainty evidence. Both follow-ups stay inside
the combined 1.5 allocated GPU-hour cap.

## Measured result

All six TC2 jobs completed normally. Total allocated time is 4,936 seconds
(82 minutes 16 seconds), within the 1.5-hour cap. The record is
[results/multiscene-validation-checks.json](results/multiscene-validation-checks.json).
No ID/OOD/physics test split was evaluated by a model.

| Validation model | Tracked observations / 5,952 | Complete all-object clips / 96 | Result |
|---|---:|---:|---|
| E0 RGB CVAE | 398 | 0 | Missing selected tracks |
| E2 numeric diffusion | 378 | 0 | Missing selected tracks |
| E3 spatial diffusion | 269 | 0 | Missing selected tracks |
| E4 auxiliary diffusion | 162 | 0 | Missing selected tracks |
| Flow CVAE | 5,654 | 80 | All eligible action/control pairs stationary |
| Flow CVAE with displacement loss | 5,765 | 80 | Reversal and changed-selection checks fail |

The original CVAE trained 20,000 updates and uses its validation-selected
16,000-update checkpoint. The flow-only checkpoint is at 20,000 updates. The
motion-supervised run also trains 20,000 updates, but combined posterior/RGB/KL
and displacement validation loss selects update 9,000. Do not describe it as
an unsupervised CVAE improvement or an independent test-set result.

The supervised follow-up has 15/16 moving forward samples but only 1/16 moving
reverse samples, 1/16 correct force-reversal pairs and 0/16 correct selected-object
pairs. Its better tracking does not establish reliable force control. At native
force, 11/16 continuous and 8/16 rendered simulator reference pairs are below the
short-window motion threshold. The separate force-0.4 selection probe passes all
16 continuous and rendered reference pairs, while both flow variants pass none.

E4's validation heads have ADE 0.1314 and timed-event F1 0.1324 at threshold 0.5.
Only 162/5,952 head/RGB object observations are jointly available; collision
agreement is undefined because no pair interval is jointly observed. The fixed
training-only collision check gives head event F1 1.0 and 247/248 tracked RGB
observations, but one missing selected observation still fails the strict gate.
It demonstrates training and sampling mechanics, not generalization.

The 16/32/64-step study uses one fixed E3 checkpoint and 64 videos per setting.
Export times are 16.25/29.37/57.59 seconds, including loading and writing samples.
Tracking is 330/220/184 of 3,968 observations; no setting has a complete selected
clip. No successful sampler was selected.

Static frame-zero copying scores SSIM 0.9446 and LPIPS 0.0622, better than the
four original models on these single-reference image metrics. This control has
no action response. The ordinary models' missing trajectories leave ADE/FDE,
trajectory spread and coverage null. Their failures remain in the record.

The [report draft](../report/README.md) embeds these comparisons and a fixed
validation sample strip. Useful validation generation, independent training
repeats and final held-out comparisons remain open. Source installation, smoke
replay and all 465 tests plus four subtests pass from the source-only ZIP.

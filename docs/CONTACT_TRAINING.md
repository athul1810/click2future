# contact-aware training follow-up

the e5 audit in pr #47 is the baseline: 22/96 clips penetrate by more than one
native pixel, 16/16 zero-force clips drift over the full horizon, and contact event
f1 is 0.2118. the earlier 0/16 drift figure covers the first half-second only.

## frozen first experiment

e6 changes the training package, keeping the e5 model architecture, observable
inputs, renderer and 16-step heun sampler. no projection, zero-force override,
postprocessing or tracker change is added. this is a separate variant and does
not isolate the effect of one loss term.

- original dataset-0.3, seed 730000: 336 training clips, 84 validation clips.
  train-only square symmetry augmentation stays unchanged. all 248 final tests
  remain reserved. no new training data is added in this first comparison.
- same initialization seed 19, width 128/four blocks, batch 64 and 6,000 updates.
  all weights start from scratch. best fixed-seed validation edm loss selects the
  checkpoint; geometry and action results do not select among checkpoints.
- each training draw assigns 25% probability to zero force, 40% to contact
  families and 35% to the remaining actions. stationary examples get an additional
  weighted clean-motion loss. there is no inference-time stationary constraint.
- a differentiable separating-axis calculation uses convex hulls from observed
  frame-0 masks. it penalizes predicted penetration beyond one native pixel and
  gap error near ground-truth contact. future poses supervise training only.
  geometric losses apply at sigma <= 0.3, where the denoising target is local.
- an acceleration-error loss supplements the existing temporal error loss.
  every coefficient is fixed in `configs/training/contact_pilot.json` before the
  allocation. temporal losses compare to labels, allowing real changes at impact.
- training is restricted to tc2 slurm. a four-update device continuation check
  must pass before the model run. the process limit is 20 minutes inside a
  30-minute allocation; interrupted attempts and source hashes are retained.

the torch geometry values are checked against the audit's independent numpy
calculation. degenerate sine/cosine values cannot shrink hulls to evade the loss.
this geometry is an approximate engineering control, not an exact physical solver.

## checks after training

run every frozen validation condition and all changed-selection controls, with
the original four seeds, image metrics and action settings. also replay every
sample pixel-for-pixel before measuring geometry. retain every condition in the
uncertainty strata, including missing tracks and undefined coverage.

the target is learned zero-force stationarity over all 32 frames, fewer deep
penetrations, and improved contact timing without losing action response or
reference coverage. report all counts even if these targets fail. one training
seed and this diagnostic corpus cannot establish generalization or a sota result.

## run

set the source/interpreter/data/revision paths as in the existing tc2 wrappers:

```sh
sbatch submission/train_contact.sbatch
sbatch submission/evaluate_trajectory.sbatch --method E6
```

training uses `C2F_DATA_DIR` and a fresh `C2F_RUN_DIR`. evaluation uses the same
data plus `C2F_TRAJECTORY_CHECKPOINT` and a fresh `C2F_TRAJECTORY_RESULTS`.

## second declared candidate

job 43598 completed the first recipe. full-horizon zero drift is 0/16 and
penetration falls to 17/96 clips, but changed-selection success regresses to
7/16 from e5's 16/16. contact event f1 is 0.3333. the unpushed object can drift
under nonzero force, so this recipe is not accepted as a replacement.

the next six-thousand-update run starts from the same seed on the same corpus.
`contact_token_pilot.json` extends stationary supervision to training tokens whose
labelled pose is unchanged, including passive objects before contact. it raises
penetration/contact weights to 256/64 and the geometric-loss sigma ceiling to
1.0. this combined training change is explicit; it does not isolate one component.
selection, uncertainty, contact timing and every failure remain in the comparison.
no zero-force or passive-object override is used during sampling.

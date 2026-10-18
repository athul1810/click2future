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

## third declared candidate: paired training actions

the second recipe (job 43600) restores early reversal and changed-selection to
16/16 and keeps zero-force full-horizon drift at 0/16. penetration remains 18/96.
both contact candidates still have passive drift in 8/32 complete forward/reverse
samples over the full horizon; e5 has 22/32. these are additional diagnostics,
not replacements for the original early checks.

the last saved interactive request is also retained: force (-0.42855, 0.25758)
on object 1. e5 reproduces the cached video pixel-for-pixel. its passive object
moves up to 9.83 pixels; the token candidate reaches 11.66 pixels. one matched
simulator realization has no collision and no passive motion. this diagnostic is
not a frozen validation case or a distributional reference.

the generator only selects object 1 for an action toward object 0. whole-scene
symmetries preserve this confound. the third candidate therefore adds a separate,
hashed training-only action supplement to the original corpus:

- exactly the existing 16 training worlds and their four existing physics
  profiles, with both objects selected, eight compass directions and force
  magnitudes 0.25/0.5: 2,048 nonzero clips plus 32 zero-force clips;
- first-frame pixels, object geometry, hidden physics and native rollout settings
  are preserved; only action changes. the original 84 validation clips and all
  248 reserved tests are untouched. no test rollout is opened;
- supplement validation rejects missing actions, changed bytes, changed physics,
  changed parent hashes or non-training membership. angular labels must replay;
- `contact_grid_pilot.json` keeps the token candidate architecture/loss/sampler,
  seed 19 and checkpoint selection rule, with 16,000 updates from scratch and a
  30-minute process limit inside a 40-minute allocation. increased data and
  updates are a combined change, not an isolated causal ablation;
- original validation, exact-pixel replay, calibrated geometry, full-horizon zero
  and passive checks, and the saved interactive request are all reported. all
  three experiments are retained even if the third candidate fails.

this expands training support, not the project's claimed deployment domain. it
cannot establish general image understanding, physical correctness or a sota win.

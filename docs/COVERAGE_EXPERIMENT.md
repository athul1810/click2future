# paired visible-coverage screen

## question and limits

E8's temporal residual barely changed generated motion and did not improve
contact. this experiment changes data while keeping the conditioned E7 model.
it tests whether adding independent visible geometry/layouts helps more than
the same number of new examples on the old visible scenes. it does not assume
data is the sole bottleneck or claim an external SOTA comparison.

the original validation has already been used for development. this is a
post-D014 exploratory screen, not a confirmatory held-out result. it cannot
promote a new default checkpoint by itself. independent training repeats and a
newly frozen holdout remain subsequent work; D014/D015 stay historical results.

## frozen intervention

both arms append 4,160 original clips, or 33,280 dihedral-augmented examples, to
the same verified 19,328-example cache. both therefore have 52,608 examples.
the 32 paired groups each use four new physical profiles, both selected objects,
eight fixed force directions, magnitudes 0.25 and 0.5, plus one zero-force case
per selected object. generation does not inspect validation or test rollouts.

| arm | additional visible scenes |
|---|---|
| narrow | cycle the original 16 training scenes twice, with fresh physical draws |
| broad | 32 independent scene seeds 810000-810031, with new geometry, appearance and layouts |

new layouts vary both initial angles, continuous pair orientation, translation
and separating-plane gap from 0.02 to 0.08 world units. shapes, dimensions,
textures and physical ranges use the existing generator ranges. separating
support planes keep the initial objects apart. original samples are unchanged.
this changes the visible scene distribution, not just its nominal group count;
geometry, appearance and layout effects are not separated in this screen.

paired rows have the same force and physical properties by original object ID.
canonical spatial slots can reorder those identities and are checked during
label preparation. the common sampling-contact stratum is the union of the two
paired training outcomes. this is privileged training-only information: it keeps
the sampling weights and paired index draws identical while the actual contacts
can differ. the actual outcomes remain recorded separately. neither evaluation
contacts nor inference physics are used for this sampling rule.

## training and evaluation

- both start from conditioned E7 EMA final update 8,000, checksum
  `93db6c714a0049b311117abe68faeef0d0e27080315bb8e67a2ecda59010c041`.
  all 1,064,580 parameters are trainable. no new model module is added.
- both use seed 19, batch 64, 8,000 updates, learning rate 1e-4, EMA 0.995,
  identical existing losses, augmentation and sample budgets. tail-risk weight
  stays zero. both reset the optimizer and private training RNG. compare fixed
  final update 8,000; validation loss never chooses a checkpoint.
- attest the original cache against the teacher before accepting the explicit
  training-data extension. original validation arrays and labels stay identical.
  do not relax the original warm-start or resume guards in existing trainers.
- retain initial tensor/RNG and sampling-weight digests, exact CUDA continuation
  fixtures, all partial states and failed attempts. incomplete arms do not form
  a matched comparison. each training arm has a 900-second bound.
- evaluate the unchanged teacher and both arms on every original validation
  condition, seed and physical reference. retain contact depth/duration and
  timing, trajectory failures, rest, both action horizons, selection, clipping,
  multireference coverage, native RGB replay and paired mass diagnostics.
- no solver, projection, clamping, reranking or changed tolerances. checkpoint
  format remains E7; the data intervention is recorded in provenance. evaluator
  source stays unchanged, so historical E8 plans remain replayable from main.

`submission/coverage_comparison.sbatch` uses one A40, four CPUs, 24 GB and at most
one hour. only scheduled TC2 jobs perform optimization. the training-only raw
generation may run locally with four CPU workers; its platform, versions and
source hashes are retained. native label replay and calibration run when the
caches are prepared. no geometry or seed is dropped after seeing a failure.

required environment: `C2F_PROJECT`, `C2F_PYTHON`, `C2F_DATA_DIR`, `C2F_CACHE`,
`C2F_WARM_START`, `C2F_COVERAGE_RESULTS`, `C2F_SOURCE_REVISION`.
`C2F_PREPARED_COVERAGE` can identify a checked retained generation. the Python
driver supports `--trained-run` for evaluation-only recovery; supply the same
`--coverage` path if generation was stored separately. all outputs must be new.

## status

implementation and data verification in progress. no trained-quality result yet.

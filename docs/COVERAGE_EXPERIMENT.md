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

the frozen generation has 1,376 actual contact clips in narrow and 1,134 in
broad; their common sampling-contact union contains 2,129 clips. equal sampling
weights do not make actual contact prevalence equal. this remains part of the
data intervention rather than an isolated test of geometry count.

stored arrays and metadata remain checksum-exact. native recipe reconstruction
allows at most eight float64 ULPs only in derived force/position vectors because
Windows and Linux trigonometry differed by one ULP in a diagonal force. the
training force must still match exactly in float32; physical labels and all
other scene fields remain exact. no scoring tolerance changes.

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

2 november: job **47305 completed**, exit `0:0`, using 1,708 allocated A40
seconds. its archive verifies and all 780 retained file digests match. both arms
finished the fixed 8,000 updates. initial tensors, private RNG and sampling-weight
digests agree. all 470 source files verify before and after the allocation.

| measure | teacher | narrow | broad |
|---|---:|---:|---:|
| complete all-object clips /96 | 96 | 96 | 95 |
| penetrating clips /96 | 11 | 9 | 15 |
| penetrating future frames | 105 | 99 | 176 |
| maximum depth, pixels | 2.4480 | 2.4895 | 2.7101 |
| longest penetration, frames | 19 | 18 | 19 |
| timed contact-event F1 | 0.3366 | 0.3366 | 0.3529 |
| failed trajectory conditions /24 | 10 | 10 | 10 |
| correct early reversal pairs /16 | 12 | 13 | 12 |
| forward selected-object spread | 0.02243 | 0.02303 | 0.02263 |

neither arm is promoted. narrow reduces the number of penetrating clips but
increases worst depth. broad increases penetration and loses one complete clip.
the additional visible distribution did not solve the existing validation task.
this does not isolate geometry count, prove broader data is harmful in general,
or establish behavior on a new holdout. physical profiles, layouts and actual
contact prevalence differ as described above.

native and independent Windows replay are pixel-exact for all 288 clips.
collision counts/events match; the largest geometric-depth difference is
2.79e-7 pixels. both CUDA continuation fixtures have equal losses and zero
model/EMA differences. full action/selection diagnostics remain supplementary to
the original scores; all exclusions and missing tracks retain their denominators.

supplementary full-horizon selection is 61/64 correct targets with 3 passive
drifts for narrow, versus 58/64 and 6 for broad (teacher: 62/64 and 2).
force scaling retains 75/76 complete comparisons in both arms: narrow increases
in all 75, broad in 74. eight selection conditions and thirteen force conditions
are excluded from full-horizon comparison because their references contact first.
these exclusions remain explicit, rather than counting them as successes.

the [complete result](results/coverage-validation.json) preserves failed cases,
all action strata, source identities and the paired response diagnostics.
the dated submission snapshot below remains historical rather than current state.

26 october submission snapshot: job **47305** is pending with dependency
`afterany:47288`, the existing AI6102 allocation. that job was left unchanged.
the coverage allocation has a one-hour cap and starts after the dependency ends.
training source is `205456f`; generation source remains `b4604d5`. all 470
training-source files verify on TC2, and the original teacher/cache and prepared
generation match their expected checksums.

634 tests and four subtests pass, with six inherited torchvision warnings.
lint and changed-source formatting pass; four inherited global-format failures
remain. the 293-file source ZIP passes inventory/link inspection,
clean-extraction smoke and 12 extracted regression tests. the real teacher and
base cache match all eight refinement bindings.

the local preflight replays labels and encodes all 66,560 augmented examples,
then verifies their observed features, hulls and object/physics identities.
initial-mask IoU is 1.0 across all 8,320 original clips. all 32 training pairs
were visually inspected. a native metadata-only probe reconstructs every scene
recipe, and full native file/label checks run again during scheduled cache
preparation. no seed was removed.

two setup failures are retained: the first upload wrapper used CRLF and failed
before extraction; the first native probe demanded exact float64 trigonometry
and found a one-ULP force difference. both occurred before optimization. the
corrected recipe guard changes no stored data or evaluation source.

[preflight evidence](results/coverage-preflight.json) records identities, label
digests and the pending-job receipt. the
[private draft backup](https://github.com/athul1810/click2future/releases/tag/untagged-81f5375e01fa3f8e2117)
contains all frozen inputs, both source archives, test logs and visual checks.
all six release assets were downloaded and their digests verified.

at submission, collecting final accounting/results, independent Windows replay
and all-seed output inspection were pending. the 2 november section supersedes
that status. no checkpoint is promoted.

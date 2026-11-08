# Contact timing from saved poses

The collision audit reads contact onsets from rendered pixels. On correct
simulator renders of the validation clips that estimator scores event F1 0.489
(`contact-reference-control.json`), so it is too noisy to judge small timing
differences. Pose models (E5-E8 and the physics baseline) also save rigid poses
for every sample (`geometry-audit/*-poses.npz`), so their contacts can be read
from geometry instead of pixels. This control asks two questions:

1. How well can contact onsets be recovered from poses when the poses are
   exactly right?
2. What does pose-based contact timing say about the saved coverage-study
   samples (Table `tab:coverage`), and how should timing be scored when hidden
   physics makes several futures valid?
3. Does the timing measure hold up on more contact conditions? The frozen D018
   replication supplies 20 on ten fresh ID scenes.

No model, threshold or recorded score changes. The D018 part reads that study's
`id_test` split after its checkpoints, plans and gates were frozen and scored.
It is descriptive only and uses the touch threshold chosen on training clips.

## Method

`scripts/pose_contact.py`:

1. builds the exact convex outline of each object from the scene specification
   (circles as 256-gons) in the audit's pixel coordinates. The outlines are
   evaluation-only ground truth, as labels are, and never a model input;
2. moves the outlines by a saved pose frame with the same frame-zero encoding
   and rotation centre as `rigid_geometry.geometry_diagnostics`;
3. computes the signed distance between the two outlines (gap, or minus the
   penetration depth) at every frame;
4. counts a contact onset in interval i when the outlines are separated at
   frame i and within the touch threshold at frame i+1, and matches onsets to
   the labels with the audit's one-to-one, one-interval event tolerance.

For the reference calibration, each training/validation clip of the frozen
dataset-0.3 corpus (03c9468e) is replayed in the simulator. The replay
reproduces the saved positions to within 1e-6 (the script refuses to continue
otherwise), and the replayed poses are encoded exactly as a model's saved poses
would be. The touch threshold is chosen on the 336 training clips only, from
0-0.5 pixels. As in the penetration control, the true outlines are fixed only
to a fraction of a pixel, so the run is repeated with the outlines placed 0.2
(primary), 0 and 0.5 pixels from the pixel-index origin.

```sh
python scripts/pose_contact.py --dataset data/frozen \
  --evaluation anchor=outputs/<anchor-validation> \
  --evaluation narrow=outputs/<narrow-validation> \
  --evaluation broad=outputs/<broad-validation> \
  --output outputs/pose-contact.json
```

Add `--held-out-dataset <D018 data> --held-out NAME=FOLDER` (repeated per arm)
to score the D018 arms. Held-out plans must have split `id_test`; validation
plans must have split `validation`. The D018 data, plans and saved poses come
from the team's checksummed `d018-frozen-replication.tar.gz` release backup.
The retained record is `docs/results/pose-contact.json` (about 8 minutes on a
laptop CPU with all six D018 arms, no GPU).

## Result 1: poses recover contact far better than pixels

Primary offset 0.2. The training sweep selects a touch threshold of 0.05
pixels (training F1 0.902). All three offsets select the same threshold and
give the same validation F1.

| Validation, 84 reference clips | Matched | Extra | Missed | F1 |
|---|---:|---:|---:|---:|
| Pose onsets vs all 44 labelled onsets | 33 | 1 | 11 | 0.846 |
| Pose onsets vs 28 frame-visible onsets | 28 | 6 | 0 | 0.903 |
| Pixel estimator on correct renders | | | | 0.489 |

The 11 missed onsets are not detector errors. Of the 44 labelled validation
onsets, 16 begin while the objects already touch at the preceding frame, and 10
are repeat contact-begin events with no visible separation since the previous
one (training: 43 of 156 and 18). Contact that breaks and resumes between two
frames 1/8 s apart cannot be seen by any frame-sampled method. Against the 28
onsets that start from a visible gap, the pose detector finds every one. Five
of its six extras there match a labelled onset one interval away, and only one
has no label at all (the single extra against all labels).

## Result 2: single-reference F1 still cannot rank the coverage arms

Applied to the 96 saved samples of each coverage-study arm:

| Event F1 vs first reference | E7 teacher | Narrow | Broad |
|---|---:|---:|---:|
| Pixel audit (Table `tab:coverage`) | 0.337 | 0.337 | 0.353 |
| Pose onsets | 0.311 | 0.274 | 0.277 |

Under the same pose rule, the exact replayed poses of the 60 valid alternative
hidden-physics references score 0.278 against the first reference (10 matched,
14 extra, 38 missed). True simulator futures score no better than the models,
confirming with an accurate detector that single-reference timed F1 cannot rank
models whose futures differ only in plausible hidden physics.

## Result 3: hidden-physics-aware first-contact timing

Instead of one reference, each sample is compared with all references for its
condition:

- occurrence agreement: the sample contacts if and only if the references do
  (scored where all references agree, which is every validation condition);
- range agreement: when both contact, the sample's first-contact interval lies
  within the references' first-contact range, plus or minus one interval;
- nearest error: intervals to the nearest reference's first contact.

The references themselves calibrate the measure: each one is scored against the
other references of its condition (leave-one-out).

| First contact | Occurrence | Timed | Within range ±1 | Mean nearest error |
|---|---:|---:|---:|---:|
| Simulator references (leave-one-out) | 84/84 (1.00) | 32 | 0.75 | 1.09 |
| E7 teacher | 0.938 | 26 | 0.846 | 1.00 |
| Narrow | 0.979 | 30 | 0.833 | 0.90 |
| Broad | 0.958 | 28 | 0.714 | 2.25 |

Median nearest error is one interval in every row. The teacher and narrow arm
place first contacts as close to the reference set as the references are to
each other. Broad samples miss more often and are further away on average.
Offsets 0 and 0.5 keep this ordering (broad range agreement 0.704 and 0.786,
mean error 2.19 and 2.11).

## Result 4: held-out D018, one-pass arm places contact worse in every seed

D018 scores six frozen checkpoints (one-pass and diffusion prior decoders,
refinement seeds 19-21) on 60 conditions in ten fresh ID scenes, four samples
each. Twenty conditions involve contact, against eight in validation. The touch
threshold stays at 0.05 pixels from the training sweep.

| First contact, offset 0.2 | Occurrence | Timed | Within range ±1 | Mean nearest error |
|---|---:|---:|---:|---:|
| Simulator references (leave-one-out) | 1.000 | 80 | 0.775 | 1.24 |
| Diffusion, seed 19 | 0.929 | 63 | 0.746 | 1.65 |
| Diffusion, seed 20 | 0.917 | 60 | 0.733 | 1.65 |
| Diffusion, seed 21 | 0.913 | 59 | 0.729 | 1.54 |
| One pass, seed 19 | 0.846 | 43 | 0.558 | 3.07 |
| One pass, seed 20 | 0.892 | 54 | 0.611 | 2.83 |
| One pass, seed 21 | 0.863 | 47 | 0.553 | 2.83 |

Occurrence is scored on all 240 samples per arm. In every seed and at every
offset, the one-pass arm misses contact more often and places first contact
further from the references: range agreement 0.52-0.61 against 0.69-0.75 for
diffusion, mean error 2.55-3.34 against 1.54-1.93 intervals. Diffusion stays
close to, but behind, the references' own agreement. Median nearest error is one
interval in every row. This confirms D018's recorded timed-contact regression
with a measure that uses all references and no pixels. Single-reference pose F1
again cannot rank: arms score 0.145-0.229 and valid alternative references
0.275.

## Limitations

- Only 8 of the 24 validation conditions involve contact (four scenes, two
  collision directions), so validation timed counts are 26-32 per row and arm
  differences there are indicative only. D018 adds 20 conditions, but all six
  checkpoints share one E7 teacher.
- Models are compared with all references of a condition, one more than each
  reference in the leave-one-out calibration. This makes the model range
  slightly wider.
- Reference first-contact ranges are wide (up to six intervals), so range
  agreement is lenient. Nearest error is the stricter measure.
- Exact outlines come from the scene specification. Circles and rectangles only;
  triangles appear only in the shape-OOD split, which is not read.
- D018 is a held-out split. It is read after every D018 decision was frozen and
  changes no gate, selection or threshold.
- This is a measurement control. It does not change any recorded score or
  promote a checkpoint, and the validation scenes were already used for
  development.

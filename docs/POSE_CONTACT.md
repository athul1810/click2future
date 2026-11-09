# Contact timing from saved poses

The collision audit reads contact onsets from rendered pixels. On correct
simulator renders of the validation clips that estimator scores event F1 0.489
(`contact-reference-control.json`), motivating a separate geometry-based timing control. Pose models (E5-E8 and the physics baseline) also save rigid poses
for every sample (`geometry-audit/*-poses.npz`), so their contacts can be read
from geometry instead of pixels. This control asks three questions:

1. How well can contact onsets be recovered from poses when the poses are
   exactly right?
2. What does pose-based contact timing say about the saved coverage-study
   samples (Table `tab:coverage`), and how should timing be scored when hidden
   physics makes several futures valid?
3. Does the timing measure hold up on more contact conditions? The frozen D018
   replication supplies 20 on ten fresh ID scenes.

Original models, scorer thresholds and historical model scores stay unchanged. The D018 part reads that study's
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

The D028 record `docs/results/pose-contact-d028.json` (about 6 minutes) uses the
D028 comparison layout:

```sh
python scripts/pose_contact.py --dataset data/frozen \
  --held-out-dataset <seed-940000 data> --held-out-study D028 \
  --held-out neural=compare/neural --held-out projected=compare/projected \
  --held-out-touch 0.3 0.5 0.8 1.0 --standoff \
  --output outputs/pose-contact-d028.json
```

## Result 1: poses recover contact far better than pixels

Primary offset 0.2. The training sweep selects a touch threshold of 0.05
pixels (training F1 0.902). All three offsets select the same threshold and
give the same validation F1.

| Validation, 84 reference clips | Matched | Extra | Missed | F1 |
|---|---:|---:|---:|---:|
| Pose onsets vs all 44 labelled onsets | 33 | 1 | 11 | 0.846 |
| Pose onsets vs 28 rule-conditioned labels (auxiliary) | 28 | 6 | 0 | 0.903 |
| Pixel estimator on correct renders | | | | 0.489 |

The all-label count retains all 11 missed onsets. Of the 44 validation labels,
16 begin while the preceding frame is within the touch band, including ten
repeat onsets without a sampled separation (training: 43 of 156 and 18).
The 28-event subset intersects true labels with the same fixed gap-to-touch
rule. At the selected threshold, recall on this subset follows by construction;
its F1 is an auxiliary rule-conditioned diagnostic. It does not independently
validate detector accuracy or establish a limit on every 8-fps estimator.


## Result 2: single-reference F1 remains sensitive to hidden-world choice

Applied to the 96 saved samples of each coverage-study arm:

| Event F1 vs first reference | E7 teacher | Narrow | Broad |
|---|---:|---:|---:|
| Pixel audit (Table `tab:coverage`) | 0.337 | 0.337 | 0.353 |
| Pose onsets | 0.311 | 0.274 | 0.277 |

Under the same pose rule, the exact replayed poses of the 60 valid alternative
hidden-physics references score 0.278 against the first reference (10 matched,
14 extra, 38 missed). True simulator futures score no better than the models,
confirming with a geometry-based detector that single-reference timed F1 mixes hidden-world variation with differences in
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
| Simulator references (leave-one-out) | 80/80 (1.00) | 32 | 0.75 | 1.09 |
| E7 teacher | 0.938 | 26 | 0.846 | 1.00 |
| Narrow | 0.979 | 30 | 0.833 | 0.90 |
| Broad | 0.958 | 28 | 0.714 | 2.25 |

Median nearest error is one interval in every row. Timing is conditional on
detected contact; occurrence is scored on all 96 model samples. Four lone
zero-force references have no leave-one-out comparator and are unscorable,
leaving 80 reference occurrence comparisons. Model timing uses four references
and reference leave-one-out uses three, so the reference row is a descriptive
sensitivity check. The paired model arms use the same number of references.


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

Occurrence is scored on all 240 samples per arm. Reference occurrence has 200 scorable comparisons; ten lone zero-force references have no leave-one-out peer. In every seed and at every
offset, the one-pass arm misses contact more often and places first contact
further from the references: range agreement 0.52-0.61 against 0.69-0.75 for
diffusion, mean error 2.55-3.34 against 1.54-1.93 intervals. Diffusion stays
close to, but behind, the references' own agreement. Median nearest error is one
interval in every row. This confirms D018's recorded timed-contact regression
with a measure that uses all references and no pixels. Single-reference pose F1
also needs this qualification: arms score 0.145-0.229 and valid alternative references
0.275.

## Result 5: the D028 projection stops objects short of exact contact

D028 applies a geometric projection after sampling the D020 neural poses: it
separates overlapping observed pixel-centre hulls (`contact_projection.py`). Its
recorded RGB contact F1 is 0.169 raw and 0.176 projected, which cannot show what
the projection does to contact timing. The raw job-52299 outputs (Athul's draft
release `d028-compare-raw-job52299.tar.gz`, `results.json` equal to the recorded
`result_sha256`) are scored here with `--held-out-study D028`. The seed-940000
corpus was regenerated locally from 314fa0c: all 210 `id_test` clip arrays match
the plan's `input_sha256` byte for byte, while metadata files differ only in the
last float digit, so the local manifest hash (75a51045) differs from TC2's.
Every reference replay passes the 1e-6 position check.

Twenty of the 60 conditions involve contact. Offset 0.2, training-selected touch
threshold 0.05 pixels:

| First contact | Occurrence | Timed | Within range ±1 | Mean nearest error |
|---|---:|---:|---:|---:|
| Simulator references (leave-one-out, 200 scorable) | 1.000 | 80 | 0.850 | 0.84 |
| Neural D020 | 0.904 | 57 | 0.491 | 3.61 |
| Neural + projection | 0.796 | 31 | 0.516 | 3.26 |

All 160 samples of no-contact conditions are correct in both arms. The drop comes
from contact conditions: 49 of 80 projected samples never reach exact contact,
against 23 raw. The projection separates observed pixel-centre hulls, and those
hulls do not coincide with the exact outlines. The recorded diagnostic
(`--standoff`, `projection_standoff_diagnostic`) replays every reference of the 20
contact conditions: at the 1,032 frames where they truly touch, the pixel-centre
hulls are still 0.23 pixels apart at the median, and anywhere from 0.91 pixels
overlapping to 0.46 pixels apart (5th-95th percentile), depending on orientation.
The closest exact gap of a projected sample has median 0.29 pixels (interquartile
-0.20 to 0.56), against -0.20 raw, so projected objects usually stop short of the
exact outline, which the 0.05-pixel touch rule counts as no contact. Offsets 0 and 0.5 give the same pattern (projected occurrence 0.783 and
0.817 against 0.900 and 0.921 raw).

A labelled sensitivity check rescores references and both arms at looser touch
thresholds (`--held-out-touch`); the training-selected threshold stays primary:

| Touch, pixels | References: range / error | Neural: occurrence / range / error | Projected: occurrence / range / error |
|---|---|---|---|
| 0.05 | 0.850 / 0.84 | 0.904 / 0.491 / 3.61 | 0.796 / 0.516 / 3.26 |
| 0.3 | 0.863 / 0.85 | 0.954 / 0.609 / 2.36 | 0.838 / 0.634 / 2.98 |
| 0.5 | 0.887 / 0.70 | 0.983 / 0.658 / 1.78 | 0.858 / 0.609 / 3.11 |
| 0.8 | 0.887 / 0.66 | 1.000 / 0.888 / 0.88 | 0.979 / 0.867 / 1.08 |
| 1.0 | 0.912 / 0.57 | 1.000 / 0.863 / 0.60 | 1.000 / 0.813 / 0.70 |

At about one pixel of tolerance both arms reach contact in every contact
condition, with timing close to the references; the projection is slightly
behind the raw sampler. At exact-contact resolution the projection leaves a
sub-pixel standoff that no pixel-level score can see. This is one checkpoint on
ten fresh ID groups and changes no D028 rule or result.

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
- D028 is likewise read after its decision was frozen. Its touch-threshold sweep
  is a sensitivity check, not a re-selected threshold.
- This is a measurement control. It does not change any recorded score or
  promote a checkpoint, and the validation scenes were already used for
  development.

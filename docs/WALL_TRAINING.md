# Matched wall-loss comparison

The saved E6-grid audit finds 32 missing observations in six wall clips. Every
missing observation loses 50.0-65.4% of its sprite pixels at the canvas. The
existing contact objective penalizes object-object penetration, with no canvas
boundary term. This experiment tests a training-only penalty for that omission.

The control and wall recipes use the same regenerated training-action supplement,
original 336 training clips, 84 validation references, initialization seed 19,
16,000 updates, balanced draw probabilities, checkpoint criterion and 16-step
Heun sampler. The two configuration files differ only in `boundary_weight`:
zero for control and 256 for wall. Both start from scratch. Prior E5/E6 weights
and reports remain separate. There is no inference correction or tracker change.

The loss transforms observed frame-zero hulls with predicted poses and penalizes
excursion beyond the native image canvas, allowing one pixel for raster error.
It applies under the same low-noise gate as the existing geometry losses. The
336 original training labels have maximum canvas excursion 0.5067 pixels in the
local unaugmented check. The allocated job also checks the complete supplemented,
augmented training set before learning and rejects a contradictory raster margin.
Validation does not choose the tolerance. This is a rendering constraint, not a
certificate of physical wall contact or conservation laws.

The initial TC2 job is capped at 50 allocated minutes, including generation, both
training attempts and full validation. Each model stops after 16,000 updates or
20 process minutes. The previous 1.5-hour pilot remains a separate completed
allocation record. No total account quota is assumed.

The planned comparison retains all 24 conditions, four sample seeds, changed
selection controls, geometry/zero/passive audits, action-family uncertainty and
wall clipping. Track failures, conditional metrics and any regression remain in
the record. All 248 final-test clips remain reserved. One seed can isolate this
switch on these examples; it cannot establish robustness across training seeds.

```sh
sbatch submission/wall_comparison.sbatch
```

Set `C2F_PROJECT`, `C2F_PYTHON`, `C2F_DATA_DIR`, `C2F_WALL_RESULTS` and
`C2F_SOURCE_REVISION` as in the source-package instructions. Use a fresh output
directory. The driver rejects recipes that differ in any other field, preserves
failed phases and releases CUDA allocations between subprocesses.

The pre-allocation source ZIP passes all 555 tests and four subtests after clean
extraction, with no skips. Geometry gradients and the paired-recipe guard pass.
All four sampled videos from each of the three legacy E6 checkpoints are
pixel-identical before and after this source change on the same CPU runtime.
Legacy config fields normalize to the disabled wall loss for trainer resume;
existing strict source/data provenance checks still apply.

Job 43788 completed both training runs, then stopped in the control clipping
diagnostic. Re-rounding a translated affine matrix changed one pixel in a
rotated sprite's native crop. Raster diagnostic version 2 extends the original
16.16 nearest-neighbour grid instead. Its native crop must still match Pillow
exactly. This follows Pillow's affine raster implementation in
[Geometry.c](https://github.com/python-pillow/Pillow/blob/12.2.0/src/libImaging/Geometry.c).
The generated RGB, tracker and existing scores are unchanged. The failed job and
its completed checkpoints are preserved; recovery evaluates those checkpoints
without training again.

## Measured comparison

Both models completed 16,000 updates on a TC2 L40S and independently selected
update 15,500 using the declared criterion. They share identical training
provenance, including action-supplement identity. Job 43788 used 816 allocated
seconds; evaluation-only recovery job 43808 used 77, for 893 GPU seconds total.
The 15-minute recovery cap kept total actual use below the original 50-minute
comparison budget. No training was repeated.

| Validation measure | Control | Canvas loss |
|---|---:|---:|
| Tracked object observations /5,952 | 5,909 | 5,951 |
| Complete all-object clips /96 | 88 | 95 |
| Incomplete wall clips /16 | 8 | 1 |
| Missing wall observations | 43 | 1 |
| Penetration above one pixel /96 | 15 | 15 |
| Full-horizon zero drift /16 | 0 | 0 |
| Full-horizon passive drift /32 | 0 | 0 |
| Early reversal pairs /16 | 16 | 16 |
| Early changed-selection pairs /16 | 16 | 16 |
| Timed contact-event F1 | 0.3400 | 0.3232 |
| Forward selected-object spread | 0.0046 | 0.0042 |

Canvas loss improves visibility on these validation cases. It does not reduce
object-object penetration or establish valid wall contact. Both models still
fail the strict tracking/selected-trajectory gate. The remaining wall failure
loses half its sprite area. Collision timing regresses slightly and forward
spread stays below reference spread 0.0258. Wall coverage is defined in three
of four scenes under the new loss, versus none for control; the fourth stays
unscorable. These are descriptive results from one training seed.

The new matched control has 88 complete clips versus the earlier A40 E6-grid
attempt's 90. Use this pair to assess the loss switch; do not substitute the
earlier control. The sampler, tracker, pixels and scores were not corrected.
Historical one-pixel-interior boundary counts remain an uncalibrated proxy,
separate from the native-canvas loss and clipping diagnostic.

The CPU recheck reproduces all 48 trajectory/contact reports, 48 future/uncertainty
reports, 12 action strata, 192 saved-pose videos and all clipping observations.
The historical 32-observation clipping record is identical under diagnostic v2.
The source-only ZIP passes 559 tests and four subtests after clean extraction.
Raw checkpoints, datasets, source archives, logs and accounting records are
backed up separately from Git and the course source ZIP.

Records: [measured results](results/wall-training-validation.json) and
[verification](results/wall-training-checks.json). All 248 final-test clips
remain reserved. Contact quality, independent repeats and final model selection
remain open in #13-#16.

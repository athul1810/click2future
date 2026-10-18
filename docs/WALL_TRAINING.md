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

One TC2 A40 job is capped at 50 allocated minutes, including generation, both
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
failed phases and releases CUDA allocations between subprocesses. Results are
pending until the complete measured comparison is collected.

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

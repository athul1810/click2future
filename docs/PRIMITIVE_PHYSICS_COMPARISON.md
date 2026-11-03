# D023: observable circle geometry versus pixel hulls

D022 did not beat D020: 22/96 validation penetrating clips versus 4/96, selected
ADE 0.02633 versus 0.02502. Keep that failed local-dynamics candidate separate
from the demo. It does not disprove the broader graph/contact approach.

The next bounded alternative follows the primitive-geometry mechanism of
[PhysGen](https://arxiv.org/pdf/2409.18964). Recognize a solid circle only when
its observed mask exactly reconstructs as a near-square Pillow ellipse. Infer
radius from the two bounding-box extents. For other masks, retain the original
convex hull and half-pixel bevel. No true geometry, scene recipe, future frame
or physical label enters generation.

Use native circle collision geometry and moment of inertia. Keep the true
angular state in the dynamics engine, but render a uniform circle without
rotating its initial pixel raster. Such a silhouette is rotationally invariant;
rotating a bitmap disk can introduce artificial corners. Subpixel centre and
radius remain ambiguous. This is an explicit physics-assisted method, not a
neural repair or the complete PhysGen pipeline.

Original validation, all 96 samples: both physics variants have complete
tracking, zero penetration and no rest/passive drift. Primitive selected ADE is
0.0238347 versus hull 0.0239263, a small improvement. RGB contact F1 is 0.2947
versus 0.2581. These opened validation results alone do not establish superiority.
The exploratory probe's first result export failed on a NumPy boolean. Preserve
that attempt. The corrected export changes only boolean serialization/output
location, not generation. Regression checks must show every production clip
matches its retained prototype and every legacy control clip remains identical.

## Prespecified fresh comparison

Freeze source and scorer before generating seed 920000. Use ten ID scene groups
and one in each other required split; score ID only. Reject overlap with the
original training corpus and visible images from opened D021. Use all six
conditions and seeds 0-3: 240 clips per method. Retain every sample and failure.

Compare the original hull baseline with the circle/hull candidate using the
unchanged RGB tracker, event proxy and one-pixel geometry thresholds. For an
additional prespecified geometric accuracy measurement, score each prediction's
relative displacement against the original scene geometry with the identical
sampled hidden world. Those true geometries enter reference evaluation only,
after both predictions are generated.

Bootstrap paired differences in same-world displacement ADE by scene group,
10,000 draws, fixed seed 20261104. This interval describes variation across ten
scenes conditional on four fixed prior draws, not calibrated latent uncertainty.

Promote the candidate as the optional demo physics method only if it retains all
240 complete tracks, zero penetration, no rest/passive drift or missing passive
samples, no worse boundary clipping, no worse first-reference selected ADE or
RGB contact F1, and its paired scene-bootstrap 95% upper bound for same-world
ADE difference is below zero. Keep the original strict single-reference gate
and all historical failures unchanged. This establishes a bounded ID comparison,
not OOD reliability, exact contact timing or successful neural physics learning.

The comparison script is `scripts/compare_primitive_physics.py`. Primitive and
legacy methods retain distinct source-based cache identities. If accepted, the
explicit `--physics-baseline` option will select the circle/hull generator first
in the demo while retaining the physics-assisted label and neural comparisons.

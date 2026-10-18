# Contact and uncertainty audit

The validation measurement work is complete. E5 still fails physical quality:
22/96 generated clips have geometric penetration above one native pixel, and
16/16 zero-force clips drift over the full video. No tracker, sampler or renderer
was changed to hide those failures. All final-test splits remain reserved.

## Measured checks

| Check | E5 | Control |
|---|---|---|
| Pose replay against saved RGB | 96/96 clips match every pixel | Exact original checkpoint and four seeds |
| Hull penetration above one pixel | 22/96 clips; 369/2,976 pair frames | 0/84 oracle-motion clips |
| Maximum hull penetration | 5.737 pixels | 0.855 pixels with oracle motion |
| Full-horizon zero-force drift | 16/16, all tracks complete | 0/16 static-reference copies |
| Original first-half-second zero-force drift | 0/16 | Original result retained |

The two TC2 audit allocations, 43514 and 43515, lasted 28 and 30 seconds.
They add no training. Geometry uses predicted poses and independent silhouettes
before renderer compositing. RGB tracking remains a separate measurement.
Oracle motion is an evaluation control and never enters learned inference.

Raw pixel overlap alone is unsuitable as a penetration test: valid oracle renders
overlap in 32/84 clips, up to 19.23% of the smaller silhouette. Both the original
raster audit and its follow-up remain recorded. Version 2 adds convex hulls of
observed pixel centres and reports separating-translation depth with a one-pixel
tolerance. This is an approximate engineering diagnostic for these convex objects.

Boundary measurements do not yet distinguish valid wall contact reliably.
They exceed one pixel in 13/96 generated and 12/84 oracle clips; maximum depths
are 3.872 and 1.385 pixels. These counts are retained rather than called a
validated wall-violation rate. Calibrate margins on training controls before
using a wall score to select a model.

## Uncertainty

All six action families remain in the analysis. Spread uses complete trajectory
pairs; coverage requires all four samples, with scored-scene counts and nulls
retained. Every noncontact family has 16/16 complete all-object clips. Contact
directions have only 12/16 and 11/16, so their coverage is incomplete.

Selected-object model/reference spread is 0.0167/0.0258 for forward motion,
0.0248/0.0258 for reversal and 0.0440/0.0429 for wall motion, in normalized image
coordinates. Similar spread does not establish calibrated uncertainty.
Zero force produces spread 0.0147 despite a stationary reference world. Its sole
reference has no pairwise-spread estimate. The full-horizon check catches late
motion that the original half-second control misses, using the same one-pixel
excursion threshold and retaining missing-track failures.

Estimated generated any-contact frequency is 0.25 for forward force and 0.1875
for zero force, versus zero in their finite simulator references. Contact-family
frequencies remain undefined when timelines are incomplete. Four samples and an
approximate contact estimator cannot certify outcome probabilities.

The [measured record](results/contact-uncertainty-audit.json) retains every
condition, bootstrap intervals, denominators, source/artifact hashes and control
results. The report embeds the tables and a fixed contact example from all four
seeds. Candidate tracker v2 still needs independent review in PR #42.

## Contact-aware follow-up

The frozen E5 findings above remain the baseline. PR #48 adds three E6 training
packages, reviewed and merged into the integration branch. Version 2 of the
passive-rest scorer counts exactly one pixel as drift, consistent with the shared
action protocol: E5, E6-first, E6-token and E6-grid score 29/32, 20/32, 8/32 and
0/32 drifting samples. Original reports remain unchanged.

Independent C review recomputes all 72 E6 future/uncertainty reports from their
saved RGB ensembles and all 18 action strata exactly. The paired-action package
has zero full-horizon zero-force or passive drift, 12/96 clips with penetration
above one pixel and contact-event F1 0.3000. All six incomplete clips are wall
cases; both contact families now have complete tracks in all 32 clips. Wall
coverage is undefined in every scene because no wall ensemble has four complete
selected trajectories.

Paired-model forward spread is 0.0040 versus 0.0258 for the finite physical
references and 0.0167 for E5. Improved stationarity therefore does not establish
useful uncertainty: forward samples remain less dispersed than the references.
Contact occurrence is 1.0 in both contact families, but the weaker event-timing
score remains visible. These measurements use one training seed and four
validation scenes. Additional training data and updates change together.

A saved-pose clipping audit explains the 32 missing object observations in the
six paired-model wall clips. Every one is an `area_change` rejection. Native
sprite masks lose 50.0-65.4% of their pixels to the canvas; all 32 padded masks
pass the existing area limits. The CLI reproduces the independently computed
observations exactly. This diagnoses clipping rather than certifying physical
wall contact. The original failures stay in the scores. See
[wall clipping evidence](results/contact-wall-clipping.json) and the source ZIP's
`click2future.evaluation.wall_clipping` command.

The [compact comparison](results/contact-uncertainty-comparison.json) keeps scene
intervals, missing counts, conditional coverage and source hashes. The report
embeds the tables and all four seeds from the same fixed E5/E6 contact case.
See [training details](CONTACT_TRAINING.md) for all retained attempts. No model is
promoted and all final ID/OOD/physics clips remain reserved.

## Reproduce and next model work

Use `click2future.evaluation.trajectory_audit` with the original plan, predictions,
checkpoint and dataset. It verifies frozen hashes, all 24 conditions and all 84
validation references, then requires pixel-exact pose replay before geometry
scoring. GPU replay uses `submission/audit_trajectory.sbatch` inside TC2.
`click2future.evaluation.uncertainty_strata` reads the saved reports; supplying
the same four artifact paths also measures full-horizon zero-force stationarity.
Exact commands are in the source-package README.

The original audit motivated the contact-aware and stationary-world training
follow-up above. Further model changes should retain that measured baseline. Keep E5's extra pose supervision
separate from the E2/E3 comparison, retain this frozen corpus as a diagnostic and
record any new training data under new hashes. Do not clamp zero-force samples,
separate overlapping objects after sampling or relax tracking to manufacture a
successful result. Useful validation motion, independent repeats and final
ID/OOD comparisons remain in T13-T15 and T17.

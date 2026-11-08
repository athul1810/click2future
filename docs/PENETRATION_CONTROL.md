# Positive control for the hull-penetration audit

The rigid geometry audit counts a clip as penetrating when the convex hulls of
the two objects' frame-zero pixel centres, moved by the predicted poses, overlap
by more than one native pixel (`rigid_geometry.geometry_diagnostics`). Its only
control so far was negative: oracle-motion renders stay below that tolerance
(0/84 clips, maximum 0.855 pixels). This control asks the opposite question: when
two objects truly overlap by a known amount, what does the audit report?

Every penetration count in the repository (`physics_screen.py`,
`trajectory_audit.py`, `prior_decision.py` and the report tables) uses this hull
depth, so the result applies to all of them.

## Method

`scripts/penetration_control.py` uses the 20 distinct frame-zero layouts among
the training and validation clips of the frozen dataset-0.3 corpus (03c9468e).
For each layout, target depth and trial it:

1. builds the exact convex outline of each object from the scene specification
   (circles as 256-gons), in the audit's pixel coordinates;
2. rotates one object by up to ±30 degrees about its observed centre, offsets it
   sideways by up to 2.6 pixels and slides it towards the other object, using
   bisection until the exact penetration depth equals the target;
3. passes that rotation and shift to the unchanged `geometry_diagnostics` as an
   ordinary predicted pose and records the reported hull depth and the raw mask
   overlap.

Targets are 0 (touching) to 4 pixels, twelve trials per layout and depth: 2,640
trials. The rasteriser fixes the true outlines only to within a fraction of a
pixel, so the run is repeated with the outlines placed 0, 0.2 and 0.5 pixels
from the pixel-index origin. Only frame-zero masks and scene specifications of
training and validation clips are read. No model, threshold or recorded score
changes.

```sh
python scripts/penetration_control.py --dataset data/frozen --output outputs/penetration-control.json
```

## Result (offset 0.2; 0 and 0.5 agree)

| True overlap, pixels | Median reported | Flagged above one pixel |
|---:|---:|---:|
| 0 (touching) | 0.00 | 0% |
| 0.25 | 0.00 | 5% |
| 0.5 | 0.23 | 20% |
| 1.0 | 0.75 | 28% |
| 1.25 | 1.01 | 51% |
| 1.5 | 1.27 | 94% |
| 2.0-4.0 | true minus 0.25 | 100% |

- The reported depth follows the true depth with slope 1.00 (1.002-1.007 across
  offsets) and reads about 0.25 pixels shallow (median error -0.25; 72% of
  trials within ±0.5 pixels; mean absolute error 0.40).
- The one-pixel tolerance therefore acts at about 1.25 pixels of true overlap.
  Overlaps of 1.5 pixels or more are caught in 94-100% of trials (0.6% missed at
  1.5 pixels or more). Touching objects are not flagged (0%, or 2% with the
  outlines placed 0.5 pixels from the origin).
- Near the tolerance the count is noisy: 14.5% of true overlaps of at most one
  pixel are flagged. Over-estimates of up to about 0.8 pixels (95th percentile)
  occur mainly for pairs that include a rectangle, whose rasterised mask extends
  slightly past its outline.
- Raw mask overlap is not a penetration measure: it exceeds one pixel for 54% of
  exactly touching pairs, consistent with the 32/84 oracle renders it flags.

## Reading penetration results

- Penetrating-clip counts are reliable for clear overlaps (1.5 pixels or more)
  and blurred near the tolerance. Small differences between methods may reflect
  overlaps close to one pixel rather than a change in clear penetration.
- Reported maximum depths are conservative by about a quarter pixel: a reported
  2.45 pixels corresponds to roughly 2.7 pixels of true overlap.
- Comparisons between methods scored with the same audit share this bias.

## Limitations

- Circles and rectangles only. Triangles occur only in the shape-OOD split and
  are not covered.
- Single frames with one rigid move. The audit replays every predicted frame the
  same way, so this covers the per-frame measurement, not motion-dependent
  effects.
- This calibrates the audit, not any model. Recorded scores are unchanged.

Record: `docs/results/penetration-control.json`.

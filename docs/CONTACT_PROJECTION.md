# D028: position-based contact projection for the neural sampler

D020 and D021 left one clear neural failure: objects pass into each other
(23/240 fresh clips in D021). Physics-guided motion diffusion such as PhysDiff
(Yuan et al., ICCV 2023) projects generated motion onto physically valid states,
and position-based dynamics (Mueller et al., 2007) resolves contact constraints
by moving overlapping bodies apart in inverse proportion to their mass.

`src/click2future/models/contact_projection.py` applies that projection to each
sampled D020 pose trajectory. In every frame, overlapping observed pixel-centre
hulls are separated along their minimum translation axis, with the sample's
prior masses as weights, and hulls are kept on the canvas. Only violations are
corrected and rotations are unchanged, so zero-force rest stays exact. The
neural model and its weights are unchanged. This is a labelled hybrid: a learned
sampler plus an explicit geometric constraint, not learned contact physics.

## Selection on original validation

Three modes were scored on the original validation split only
(`docs/results/contact-projection-selection-d028.json`). The rule was the lowest
selected plus other-object same-world ADE among modes with zero penetration. The
per-frame, mass-weighted mode was chosen. Carrying corrections forward was worse.
Keeping the selected object fixed lowered its own error but raised the other
object's by 19%, so it was not chosen. Two earlier validation runs are recorded
in that file: a one-pixel wall margin over-corrected wall contacts, and a run
without a 0.001 px float margin left six boundary clips from float32 rounding.

## Fresh test (TC2 job 52299)

The mode, rules and margins were committed at 314fa0c before generating seed
940000. Frame-zero overlap with the training, D021, D023 and D026 corpora is
rejected. Rules: 240/240 complete, zero penetrating clips, no more boundary
clips, no rest or passive drift, contact F1 not lower, selected ADE at most
0.0005 higher, and a paired scene-bootstrap 95% upper bound for same-world ADE
below 0.0005. The margins are non-inferiority limits: the projection exists to
remove penetration, and D024 suggests that prior sampling accounts for much of
single-reference ADE.

| Fresh D028 ID | Neural D020 | Neural + projection |
|---|---:|---:|
| Complete clips | 240/240 | 240/240 |
| Penetrating clips over 1 px | 19 | 0 |
| Boundary-clipped clips | 25 | 0 |
| Rest / passive drift | 0/40, 0/80 | 0/40, 0/80 |
| RGB contact F1 | 0.169 | 0.176 |
| First-reference selected ADE | 0.02495 | 0.02476 |
| Same-world selected ADE | 0.00706 | 0.00712 |
| Same-world other-object ADE | 0.00320 | 0.00355 |

Paired same-world interval: [-0.00014, 0.00028]. All rules pass, so the
projection is accepted as the contact layer for the neural sampler. The other
object's same-world error rises by 11%, which the rules do not cover and is
reported here. The physics-assisted baseline remains more accurate under the
same hidden physics (D021: 0.00348 versus 0.00711 same-world ADE). Ten fresh ID groups, one
checkpoint and no OOD claim. Record: `docs/results/contact-projection-d028.json`.

## D030: simulator reference controls (TC2 job 52309)

The same corpora were re-rendered by the simulator and scored unchanged.
`true_world` uses the true hidden physics: a reference control for the scorer,
not a mathematical ceiling, since the scorer itself is imperfect. `sampled_world`
uses the four prior draws: a prior-sampling control, not the best any method
could reach. ADE is in normalized units; 0.025 is about 1.6 native pixels.

| Corpus | Generator | Selected ADE | Pixels | Contact F1 |
|---|---|---:|---:|---:|
| D026 | Simulator, true physics | 0.00447 | 0.29 | 0.590 |
| D026 | Simulator, sampled physics | 0.02189 | 1.40 | 0.184 |
| D026 | Hull 0.5 px / 0.15 px | 0.02206 / 0.02213 | 1.41 / 1.42 | 0.147 / 0.208 |
| D028 | Simulator, true physics | 0.00418 | 0.27 | 0.250 |
| D028 | Simulator, sampled physics | 0.02457 | 1.57 | 0.172 |
| D028 | Neural / neural + projection | 0.02495 / 0.02476 | 1.60 / 1.58 | 0.169 / 0.176 |

Selected ADE for every prior-sampling method sits within 1-2% of the sampled
physics control. This supports a hidden-physics explanation for much of the
remaining error, but does not isolate all of it.
The RGB contact F1 of the simulator's own renders is 0.25-0.59 depending on the
corpus, so that metric is a weak, corpus-dependent proxy and small F1 changes
should not be read as contact-timing improvements.
Record: `docs/results/metric-ceilings-d030.json`.

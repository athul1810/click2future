# Report draft

`main.tex` is Group 7's working AI6132 technical review. It includes the
E0/E2/E3/E4 validation comparison, SSIM/LPIPS, auxiliary head-versus-RGB scores,
the measured 16/32/64-step sampler study and the CVAE decoder follow-ups.
It now also includes the separate E5 trajectory model, its contact/uncertainty
audit, the full-horizon zero-force failure and an embedded contact example from
all four seeds. The draft also includes all three E6 contact-aware attempts, the
version 2 passive rescore, E6 uncertainty by action family and a fixed E5/E6
four-seed comparison, and the matched canvas-loss study. It keeps conditional metrics, failed controls and
oracle-calibration limits explicit.
The original methods fail useful validation generation. This is not the final
submission PDF or a successful physical-world model claim.

Build from the repository root with pdfLaTeX:

```sh
mkdir -p outputs/report
pdflatex -interaction=nonstopmode -halt-on-error -jobname=group_07_report_draft -output-directory=outputs/report report/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=group_07_report_draft -output-directory=outputs/report report/main.tex
```

The output is `outputs/report/group_07_report_draft.pdf`. Keep the final
`group_07_report.pdf` separate until experiments and team review are complete.
The embedded frame strip uses the first sorted validation forward condition,
seed zero and six fixed times. It shows prior/EMA samples and one simulator
reference. Historical one-clip figures remain in the source history.

`evidence.json` binds report, figure and measured result hashes. The main pilot
uses 16 training scenes and four validation scenes, with four dependent samples
per condition and one training seed per method. Image confidence intervals
resample scenes; they do not cover training-seed uncertainty. Tracker version 2
was reviewed by Sanjay and merged in #42. Its results remain separate from
historical version-1 scores.

The bibliography contains no external links. Member-reviewed contributions in `contributions.md`
and the final submission checks remain open.
Working deadline: 14 November 2026.

The final held-out evaluation (D014, job 46716) and the secondary E2/E3 test
result at E0's training-clip count (D015, job 46837) are in Sections 3.15-3.17, with Tables 10-13
and Figures 4-5. Section 3.15 ends with the E0 held-out analysis. Every table value is checked against
`docs/results/final-evaluation.json` and `docs/results/final-evaluation-d015.json`.
Figure 4 uses the first sorted ID-test forward condition and Figure 5 the first
sorted forward condition of each OOD split, with sample seed 0 and training seed
0. All 14 videos match the job prediction manifests.

The E6 uncertainty evidence is independently recomputed from all 72 saved RGB
ensembles and all 18 action strata. The compact comparison is
`docs/results/contact-uncertainty-comparison.json`. That audit opened no final test split.

The reproducible wall-clipping diagnostic explains all 32 missing E6-grid wall
observations while preserving their failed video scores. Results are in
`docs/results/contact-wall-clipping.json`; the CLI is included in the source ZIP.

The matched training-only canvas penalty improves complete clips from 88/96 to
95/96 without fixing penetration or overall strict quality. See
`docs/results/wall-training-validation.json`. Both completed checkpoints and the
failed diagnostic attempt are retained; recovery evaluates without retraining.

The collision audit adds reference-video calibration and all sustained-overlap
failures. Its timing scores retain the original estimator, including its errors
on correct simulator videos. The current E6 force-range mismatch is recorded
before opening force-OOD tests. See docs/COLLISION_VALIDATION.md.

The 2 November review export has 26 pages. Section 4 records E7 shared-world
refinement, E8 temporal memory and the completed paired coverage study. These
are post-D014 experiments on reused validation scenes, with one training seed;
none is promoted. All new table values come from the retained result records.
The export has no external URLs, link annotations, undefined references or
overfull boxes. The one-page contribution export keeps Ananya's reviewed status
and the other three pending confirmations. The editable presentation and replay
are review artifacts, separate from the three required upload files.

## fresh ID replication, 3 november

The current review export is 29 pages. D018 retains all six frozen models,
ten new ID scene groups and every failed gate. The contact improvement repeats
across three refinement seeds conditional on one teacher; it is not an overall
quality or external SOTA win. See `docs/results/prior-replication.json` and
`docs/PRIOR_REPLICATION.md`. The figure shows the first sorted condition with
every sample/reference row, and the contribution draft includes the latest work
while keeping three member confirmations pending.

# Final review, 12 November 2026

The implementation through PR #94 is merged at `75854bc`. No new model is
promoted by this report refresh. Completion remains targeted for 14 November.

## Verified package

- Report: 34 pages, from the supplied Typst archive, including D031-D039, component comparisons, negative results
  and the D028 contact-standoff analysis. D037 contributes no numeric evidence.
- Contribution statement: two pages of LaTeX, with all four accounts recorded. At Shrijeet's request,
  his existing paragraph was verified against nine authored commits and ten
  retained evidence files. Later work is separately attributed from merged history.
- Source ZIP: 419 files; deterministic rebuild, clean extraction and procedural
  replay smoke check pass. Datasets, weights and external links are excluded.
- Runtime at `75854bc`: 832 tests and four subtests pass on Windows; two
  TC2/Linux-only checks are skipped. Lint and formatting pass.
- Eight source-package diagnostic drivers import, and the trajectory sampler
  exposes the explicit momentum and projection flags. Defaults are unchanged.

The report's source hashes use LF-normalized UTF-8 for text and exact bytes for
binary assets. The archive manifest continues to hash its exact stored bytes.
The newer training results have teammate-reviewed TC2 reproductions. This
refresh checks the compact evidence and source; it does not claim a new
independent replay of all raw D031-D039 outputs.

## Model and demo boundaries

D035 improves a narrow matched-world measure across three training seeds.
D038 improves aggregate matched-world ADE by 16.8% on one fresh ID study;
most of that gain comes from the explicit momentum block. Contact F1 and
collision-family ADE regress. D036 selection is worse than a fair uniform
control. D039 is an opened-corpus, short exploratory screen. The original
strict gate still fails. These results do not establish calibrated uncertainty,
general physical reliability or superiority to external state-of-the-art models.

The portable photograph demo is recorded replay with two saved opposite actions.
It shows all eight native samples and explicitly rejects uncached requests.
Fresh photograph generation needs an allocated TC2 worker; its two measured
four-sample requests took about 133 seconds each. The external pretrained model
and development photograph are separate from the controlled-model evaluation.

## Remaining external steps

1. Apply the edited `main.typ`, `references.bib` and new native photograph figure
   to the shared Typst project. The supplied archive was reconciled and compiled
   locally; no cloud edit is claimed. Complete the group's final report review.
2. Check the live NTULearn instructions, coordinate the single group upload with
   Sanjay, and retain its receipt. Submit `group_07_report.pdf`,
   `group_07_contribution.pdf` and `group_07_code.zip`. The possible 16 November
   cutoff does not move the 14 November target.

The World Models paper presentation is separate. Click2Future slides, live demo
operation and a team demo rehearsal are optional and are not submission gates.

## Rebuild

Compile the assessed report with `typst compile report/main.typ group_07_report.pdf`
from the repository root. It uses `report/references.bib` and the existing figure
folder. Compile `report/contribution.tex` twice with `pdflatex`, setting the job
name to `group_07_contribution`. The earlier `report/main.tex` is retained as an
alternative source snapshot; do not submit two report versions. Build and inspect
the code ZIP with the commands in SUBMISSION.md.

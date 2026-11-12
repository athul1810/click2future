# Final review, 12 November 2026

The implementation through PR #94 is merged at `75854bc`. No new model is
promoted by this report refresh. Completion remains targeted for 14 November.

## Verified package

- Report: 41 pages, including D031-D039, component comparisons, negative results
  and the D028 contact-standoff analysis. D037 contributes no numeric evidence.
- Contribution statement: all four accounts recorded. At Shrijeet's request,
  his existing paragraph was verified against nine authored commits and ten
  retained evidence files. Later work is separately attributed from merged history.
- Presentation: 20 editable slides, with D038 and a follow-up summary. The
  recorded photo montage retains every native paired-action sample.
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

1. Reconcile the shared Typst project with these report sources. The current
   automation session exposes no browser or connected document; the cloud
   project has not been read or edited. An exported project ZIP and PDF are
   sufficient for an exact offline comparison.
2. Run the complete presentation on the presentation machine with all speakers,
   including embedded playback and the portable recorded fallback. Automated
   browser checks are not a team rehearsal.
3. Check the live NTULearn instructions, coordinate the single group upload with
   Sanjay, and retain its receipt. Submit only `group_07_report.pdf`,
   `group_07_contribution.pdf` and `group_07_code.zip` unless the course changes
   the requirements. The possible 16 November cutoff does not move our target.

## Rebuild

From the repository root, run `pdflatex -interaction=nonstopmode -halt-on-error
-jobname=group_07_report -output-directory=<existing-output-folder> report/main.tex`
twice. Keep `report/followups_20261112.tex` and the figure directory beside the
main source. Build and inspect the code archive with
`python -m click2future.submission build --output <folder>/group_07_code.zip`
and `python -m click2future.submission inspect --archive <folder>/group_07_code.zip`.

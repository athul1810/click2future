# Report draft

`main.tex` is Group 7's working AI6132 technical review. It includes the
E0/E2/E3/E4 validation comparison, SSIM/LPIPS, auxiliary head-versus-RGB scores,
the measured 16/32/64-step sampler study and the CVAE decoder follow-ups.
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
resample scenes; they do not cover training-seed uncertainty. Candidate tracker
version 2 results remain separate from historical version-1 scores.

The bibliography contains no external links. Member-reviewed contributions,
useful model quality, independent repeats, reserved ID/OOD tests and the final
submission checks remain open. Working deadline: 14 November 2026.

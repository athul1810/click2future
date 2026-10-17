# Report draft

`main.tex` is a working technical-review draft for Group 7. It describes the
implemented pixel diffusion and CVAE, evaluation protocol, simulator calibration
and measured pilot failures. It does not certify the final course experiments.

Build from the repository root with pdfLaTeX, including the standard geometry
and AMS packages:

```sh
mkdir -p outputs/report
pdflatex -interaction=nonstopmode -halt-on-error -jobname=group_07_report_draft -output-directory=outputs/report report/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=group_07_report_draft -output-directory=outputs/report report/main.tex
```

The output is `outputs/report/group_07_report_draft.pdf`. Keep this separate from
the final `group_07_report.pdf` until the remaining experiments and team review
are complete. The two embedded figures show actual GPU-generated samples, with
simulator reference rows. Bibliography entries contain no external links.

Figure and run hashes are recorded in `evidence.json`. The pilot uses one training
clip and one validation condition. Four seeds are dependent samples; they are not
four independent test scenes. Both methods fail the declared video-quality gate.
Final E0/E2/E3 results, the validation study, component/generalization/uncertainty
analysis and reviewed contribution descriptions remain open.

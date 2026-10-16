# Submission guide

Source: `course project technical review.pdf`, pages 4-5 and 8, and the course submission instructions.

## Files to submit

Submit these three files for **Group 7**, using `07` in the filenames.

| Filename | Contents |
|---|---|
| `group_07_report.pdf` | Main technical report, at least four pages |
| `group_07_contribution.pdf` | Each member's actual contribution |
| `group_07_code.zip` | Source-code package |

One group makes one submission through **NTULearn > course site > Assignments > submission link for Project 1**. The PDF allows replacement submissions without a stated limit. Coordinate uploads through PRAMOD SANJAY, the first member in the supplied roster.

**Internal completion deadline: 14 November 2026.** All three deliverables and the
presentation must be ready then; freeze artifacts on 13 November. Page 5 of the
brief lists 16 November for upload. Confirm the exact NTULearn cutoff; this does
not change the internal completion date.

## Report

- At least four pages, organized as introduction, methods, experiments and analysis, and conclusion.
- Compare at least two methods on at least one dataset using the team's own experiments.
- Include hyperparameter and component analysis, strengths and limitations, visualizations, and future extensions.
- Put tables, figures, explanations, and references inside the PDF. Do not rely on a website, notebook, demo link, or cloud folder for assessed material.
- Use normal numbered or author-year citations. For the submitted PDFs, write bibliographic entries without external URLs or clickable external links.

Use [REPORT_OUTLINE.md](REPORT_OUTLINE.md) to draft the report. The PDF gives no page maximum or prescribed font/template; check for additional instructions on NTULearn. Plan at least four pages of substantive report content rather than relying on references to reach the minimum.

## Contribution PDF

State the group ID and every member's full name as shown on the student card. Describe what each person actually implemented, evaluated, analyzed, or wrote. Include code/module names, experiment IDs, or report sections as evidence without external links.

Use [CONTRIBUTION_TEMPLATE.md](CONTRIBUTION_TEMPLATE.md). Each member should review their contribution description before the PDF is exported.

## Source-code ZIP

Include the source needed for the project: simulator and data-generation code, model definitions, training and evaluation code, relevant app code, configs, dependency files, and local setup/run instructions.

Exclude:

- Datasets, generated training/test clips, and copied sample datasets.
- Trained models, pretrained weights, checkpoints, and serialized model binaries. Model architecture source code is part of the source code.
- External links, model/dataset download links, and cloud-storage pointers used instead of submitted material.
- Git history, local environments, caches, secrets, logs, and working files that are not source code.

Build the ZIP from an explicit list of allowed files. Do not ZIP this entire repository: the working README, task board, research references, and experiment manifests contain GitHub and other external links. Prepare local submission instructions with no external URLs. Keep required third-party notices and comply with their license terms when choosing files to include.

Our dataset is procedural, so include the generator, fixed configs, and seed/split-generation logic instead of the data itself. Include training code instead of model weights. Test extraction and CPU checks without existing datasets/checkpoints; document that reproducing trained results requires generating data and training on the provided GPU.

Code comments are optional according to page 5. Existing useful comments do not need to be removed.

## Final check

- [ ] Verify all three filenames use `group_07_` and coordinate the upload with PRAMOD SANJAY.
- [ ] Open both PDFs and inspect every page for readability, missing figures, and broken formatting.
- [ ] Verify the main report has at least four pages and covers every required analysis.
- [ ] Confirm every member appears in the contribution PDF with a reviewed description.
- [ ] Inspect the ZIP for datasets, model files, external links, and accidental working files.
- [ ] Extract the ZIP in a clean folder; verify source imports, documented commands, and locally generated smoke inputs.
- [ ] Ensure assessed results are included in the report and require no external link.
- [ ] Upload the three files as one group submission and record the NTULearn receipt privately.

Slides and a separate demo video are not required uploads. Keep optional presentation material in the team's working files unless the course later requests it.

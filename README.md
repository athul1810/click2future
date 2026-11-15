# Click2Future

Select an object in a tabletop image, draw a force arrow and compare four
generated futures. Click2Future is Group 7's AI6132 technical-review project.

We built a procedural dataset, trained CVAE and video-diffusion baselines, and
studied how action conditioning, pose supervision and physical corrections
affect the generated motion. The interactive comparison supports synthetic
images and a separate photograph interface using externally pretrained models.
The report contains our measured results and the limits of each comparison.

## Results and scope

The main held-out comparison uses three training seeds per method and five
splits: ID, shape, texture, force and repeated hidden physics. The flow CVAE
preserves appearance and follows push direction over the full clip at the
training force, with unreliable magnitude and target selection. Longer pixel-diffusion training
removes noise, but identity changes prevent a reliable physical comparison of
its numeric and spatial conditioning.

Pose-supervised models improve tracking and action response. Contact-aware
training improves rest behaviour, and a canvas loss improves wall visibility.
Separate pose-model studies test sampled physical properties, decoder choice,
geometric projection and dynamics losses. The D038 hybrid reduces same-world
trajectory error by 16.8% on ten fresh ID scenes, mainly through explicit
momentum correction. Collision error increases, and distributional scores still
favour a prior-matched simulator. The improvement applies to that matched ID study.

| Comparison | Models | Interpretation |
|---|---|---|
| RGB generation | E0 flow CVAE; E2 numeric and E3 spatial pixel diffusion | Main held-out comparison; RGB supervision |
| Validation ablations | RGB CVAE, E1 action-blind control, E4 auxiliary heads | Diagnose decoder, conditioning and supervision choices |
| Pose generation | E5/E6 and E7-E11 follow-ups | Additional pose/contact supervision; separate from the RGB comparison |
| Physical corrections | Projection, dynamics and momentum studies | Explicitly labelled hybrids with matched controls |
| Photograph interface | Force Prompting and StreamForce | External pretrained models; excluded from controlled-model scores |

Core videos contain two rigid objects, 32 frames at 8 fps and 64 by 64 pixels.
Synthetic uploads require two separated visible objects. Their masks come from
pixels; hidden properties and future labels are unavailable at inference.
Exact simulator references require scene metadata. Changing a reference world's
friction does not change model conditioning.

## Run the project

Follow [development setup](docs/SETUP.md) for the dependency lock, environment
check and CPU tests, or [the TC2 workflow](cluster/README.md) for scheduled GPU
training. Reproducing trained results requires generating data and training;
datasets and checkpoints are not part of the source submission.

| Task | Instructions |
|---|---|
| Generate and replay data | [Simulator and data contract](docs/ARCHITECTURE.md) |
| Train or evaluate models | [Training](docs/TRAINING.md) and [metric usage](docs/METRICS.md) |
| Inspect the synthetic interface | [Demo](app/README.md) |
| Read the assessed report sources | [Report](report/README.md) |
| Build and check the source archive | [Submission guide](docs/SUBMISSION.md) |

The source archive has its own standalone instructions and CPU smoke check.
They generate temporary inputs without requiring saved data or models.

## Group 7

| Member | Work |
|---|---|
| PRAMOD SANJAY | Simulator, data and physical training objectives |
| KAPOOR ANANYA | CVAE experiments, action diagnostics and contact measurement |
| MAITI SHRIJEET | Diffusion, pose-model follow-ups and interactive interfaces |
| BOBAN ATHUL KRISHNA | Baselines, evaluation and physical-quality studies |

[The contribution statement](report/contributions.md) records the implementation,
experiments, analysis and writing in more detail. Evaluation is shared by Athul
and Ananya; each member contributed analysis for their own experiments.

## Submission

Submit exactly `group_07_report.pdf`, `group_07_contribution.pdf` and
`group_07_code.zip`. The two PDFs contain the assessed material; the ZIP contains
source, configs and tests, without datasets, model weights or external links.

Athul checked NTULearn on 14 November: the cutoff is **16 November 2026,
23:59 Singapore time (UTC+8)**, with unlimited attempts. Coordinate one upload
and retain the receipt. Prepared files and a GitHub release do not constitute a
portal submission. The World Models presentation is a separate assessment.

## Working records

[Issues](https://github.com/athul1810/click2future/issues) track current work.
[Results](docs/results/) retain compact experiment records;
[decisions](docs/DECISIONS.md) and dated plans explain earlier choices.
Historical plans describe the scope at the time they were written. Keep full
data, weights, videos and local research notes outside Git.

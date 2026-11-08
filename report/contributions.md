# Contribution statement draft

Group 7. Working deadline: 14 November 2026.

This draft records work supported by repository history and saved experiments.
Each member must review their description and add any completed work not recorded
here before the contribution PDF is exported. Commit counts do not measure effort
or establish contribution shares.

| Member | Recorded completed work | Evidence | Member review |
|---|---|---|---|
| PRAMOD SANJAY | Implemented the seeded tabletop physics simulator, renderer and force/action contract; built deterministic rollouts with physical labels, replay, dataset generation, loading, validation and leakage-safe splits. Added dataset-0.2 with controlled directional, object-collision and wall-contact cases and contract checks. Implemented the merged contact-dynamics training experiment with optional contact-transition and normal-velocity losses and matched control/candidate configurations. | `sim/backend.py`, `sim/rollout.py`; commits `7530d19`, `0a22a38`, `5b96e4b`, `134db03`; merged PR #80 (`64c8bf0`). | Reviewed |
| KAPOOR ANANYA | Set up TC2 on a second account and added the macOS/Linux setup helper; independently reproduced the RGB and flow CVAE pilots; ran the CVAE β × latent-size validation study; added and measured the CVAE force probe, full-horizon motion timing and selection-drift diagnostics, which trace the flow CVAEs' "stationary" verdict to the 0.5-second action window and their missing force scaling and unselected-object drift to training force and selection coverage; wrote report subsection 3.7 and the study evidence record; reviewed the collision-audit and evaluation-workload change, adding a label-only hidden-physics control. | `scripts/tc2_setup.sh`, `cluster/cvae_study.sbatch`, `submission/cvae_evaluation.py`, `submission/cvae_force_probe.py`, `submission/cvae_motion_timing.py`, `submission/cvae_selection_drift.py`, `report/main.tex` (3.7); `docs/results/cvae-study-checks.json`; TC2 jobs 46286, 46302, 46303, 46333; commits `37761d7`, `ca8b1c9`, `a768b5c`, `83bb48c`, `12f574b`. | Reviewed |
| MAITI SHRIJEET | Built diffusion training/sampling and action conditioning; implemented the interactive synthetic-image comparison and separately credited photograph extension with allocated worker, provenance, cancellation/recovery and replay checks. Ran and documented E5/E6 contact/stationarity, shared-world, memory, coverage, surface and prior-decoder studies, fixed-teacher repeats and the fresh D018 evaluation. Independently replayed retained model outputs, backed up native results and prepared report/presentation packages. | `training/train_diffusion.py`, `models/object_motion.py`, `training/train_contact.py`, `demo/`; trajectory/contact/shared-world/memory/coverage/surface/prior result records; commits `df2e9d7`, `5c21e30`, `4d5c903`, `8b117c7`, `f1238eb`. | Pending |
| BOBAN ATHUL KRISHNA | Implemented the CVAE baseline, generated-video evaluation, tracker calibration, selection controls, E4 and the uncertainty/contact audits, and packaged the source; ran the E10 routing comparison, the D017 held-out replication and the D019-D030 physical-quality tests with the physics-assisted baseline on TC2, and reviewed teammate changes and the report draft. | `training/train_cvae.py`, `evaluation/`, `submission.py`, `report/main.tex`; multiscene, contact/uncertainty and canvas-loss result records; commits `9b92c07`, `9084ae0`, `d007086`, `e4374bb`, `76132ed`, `7433868`, `bc77c7d`; PRs #76 and #78, with the D019-D030 result records. | Reviewed |

The final statement should describe shared analysis, discussion, review and writing
from the members' reviewed accounts. This is not a percentage allocation or the
final `group_07_contribution.pdf`.

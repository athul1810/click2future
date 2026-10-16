# Click2Future source, Group 7

This archive contains the current simulator, data generator, denoiser prototype,
auxiliary losses, evaluation code, configs, and correctness checks. It contains no
datasets or model weights. Dataset training, diffusion sampling, the CVAE baseline,
and the demo are still unfinished in this snapshot. This is not the final submission.

Use Python 3.12. Create a fresh environment and install the project from this folder:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints.txt -e '.[cpu,dev]'
.venv/bin/python submission/smoke.py
.venv/bin/python -m pytest -q
```

Use the matching CPU PyTorch distribution for CPU checks. TC2 training uses the
cluster's supported GPU environment. Dependency indexes can be configured outside
this archive through pip's normal environment or site configuration. The constraints
pin unambiguous versions from the working dependency lock. This is not an offline
dependency bundle. Package installation may need network access or a package cache.

The smoke command generates its own temporary two-variant dataset, validates replay,
loads a batch, and checks that every imported project module comes from this folder.
No existing dataset, checkpoint, GPU, or pretrained metric weights are needed.
On Windows, use `.venv\Scripts\python` for the Python commands.

Generate a grouped dataset with reproducible seeds and scene-level splits:

```sh
python -m click2future.cli generate-dataset --output-dir data/smoke --group-count 4 --variants-per-group 2 --seed 0 --config configs/simulator/smoke.toml
python -m click2future.cli validate-sample --input-dir data/smoke --sample-id scene-00000000-variant-000 --replay
```

The generator prints its manifest location. Read the generated dataset manifest for
the exact sample IDs. Keep generated data and all model files outside the submission.

Evaluation modules provide `--help` for their input contracts:

```sh
python -m click2future.evaluation.video --help
python -m click2future.evaluation.images --help
python -m click2future.evaluation.futures --help
python -m click2future.evaluation.aggregate --help
python -m click2future.training.check_auxiliary --help
```

Image metrics need the `image-metrics` extra. LPIPS also needs pretrained AlexNet
weights in the user's Torch cache. Weights are excluded from this archive. The smoke
check and the core simulator/trajectory tests do not require them. Training and
sampling commands must be added and verified before preparing the final code ZIP.

`MANIFEST.json` lists each included file and its SHA-256. The code ZIP is one of the
three required submission files, alongside `group_07_report.pdf` and
`group_07_contribution.pdf`.

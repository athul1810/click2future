# Development setup

This setup covers local Python development, notebooks, and environment diagnostics.
The simulator, datasets, trained models, training/sampling commands, evaluation, and
demo are not implemented yet. An untrained denoiser prototype and synthetic profiler
exist. TC2 training validation remains pending T02 access and hardware details.

## Environment and dependencies

- Python **3.12.15**, pinned in `.python-version`; uv downloads it when needed.
- uv **0.13.0**, required by `pyproject.toml`.
- PyTorch **2.11.0** for both backends: `cpu` or CUDA 12.8 (`cu128`).
- NumPy, Pillow, and PyYAML for numerical/image IO and YAML configurations.
- pytest, Ruff, and ipykernel in the default `dev` dependency group.

`uv.lock` records exact direct/transitive versions, sources, and artifact hashes.
CPU and CUDA extras are mutually exclusive. The CUDA option supports Windows/Linux
with a compatible NVIDIA GPU and driver; it does not certify the TC2 environment.
Backend configuration follows [uv's PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Windows / PowerShell

From the repository root, using an existing Python 3.12 installation with the `py` launcher:

```powershell
py -3.12 -m pip install --user uv==0.13.0
py -3.12 -m uv sync --locked --extra cpu
.\.venv\Scripts\python.exe -m click2future.env_check --device cpu
```

To use an NVIDIA GPU instead:

```powershell
py -3.12 -m uv sync --locked --extra cu128
.\.venv\Scripts\python.exe -m click2future.env_check --device cuda
```

These commands use the environment interpreter directly and do not require PowerShell
activation or changes to execution policy. Always select the intended extra when
syncing; plain `uv sync` installs neither PyTorch backend. Likewise, if using `uv run`,
repeat `--locked --extra cpu` or `--locked --extra cu128` so it retains the chosen backend.

## Linux CPU setup

With an existing Python/pip installation:

```bash
python3 -m pip install --user uv==0.13.0
python3 -m uv sync --locked --extra cpu
.venv/bin/python -m click2future.env_check --device cpu
```

If the distribution forbids user pip installations, install the pinned uv release
using an [official installation method](https://docs.astral.sh/uv/getting-started/installation/),
then use `uv sync --locked --extra cpu`. The Linux commands are provided for teammates;
local validation was performed on Windows. Do not change the cluster's system Python,
driver, or CUDA modules. Record its constraints in `COMPUTE.md` before selecting a
compatible training environment and running an explicit CUDA check on TC2.

## Verification

```powershell
.\.venv\Scripts\python.exe -m click2future.env_check --device auto --output outputs\environment.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
py -3.12 -m uv pip check --python .\.venv\Scripts\python.exe
git diff --check
```

On Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`, use `/`
in output paths, and invoke uv as `python3 -m uv` or `uv`. The installed
`click2future-check` entry point runs the same diagnostic.

The diagnostic reports interpreter/package versions and device details, checks PNG,
NumPy archive, YAML, and NumPy/PyTorch roundtrips, then exercises a convolution's
forward/backward pass, optimizer update, and exact checkpoint reload on CPU and,
when selected, CUDA. An explicit unavailable CUDA request returns exit code 1.
These checks validate the environment, not physics or model quality.

The two initial tests cover CPU report creation and the explicit unavailable-CUDA
failure. Add simulator/model/evaluation tests with those implementations.

## Artifacts and configuration

### Profile the untrained architecture

```powershell
.\.venv\Scripts\python.exe -m click2future.profile_pilot --device cuda --output outputs\pilot-profile.json
```

This uses `configs/pilot.yaml` and synthetic tensors. Use `--auxiliary` for the optional
heads and `--batch-size`, `--future-frames`, or `--base-channels` to test dimensions.
Measure on an uncontended GPU; inspect the per-step spread and `timing_stable` rather
than turn a short unstable run into a training-hours promise. CPU/GPU correctness and
memory evidence are separate from learned video quality. See [BUILD_PLAN.md](BUILD_PLAN.md).

The initial tests now also cover the denoiser's shape/gradient/information controls,
CPU optimizer/RNG restart continuity, and refusal to profile without a requested Slurm allocation.
Follow [the TC2 guide adaptation](../cluster/README.md) for cluster execution.

`configs/development.yaml` defines repository-relative artifact locations:

| Directory | Intended contents |
|---|---|
| `data/` | Generated local datasets |
| `checkpoints/` | Model/checkpoint artifacts |
| `outputs/` | Environment reports and generated outputs |
| `runs/` | Run logs and resolved configurations |

These directories, `.venv/`, and `.vscode/` are ignored by Git. The development config
is a template; no training command consumes it yet. Use YAML for future configurations,
JSON for reports/metadata, and NumPy `.npz` plus JSON as the initial archive approach.
Finalize dataset semantics and simulator/backend choices in T04 before implementation.
Never store credentials in tracked configurations or reports.

## VS Code and notebooks

Open the repository folder, or use the local workspace:

```powershell
code .\.vscode\click2future.code-workspace
```

The local workspace selects `.venv\Scripts\python.exe`, configures pytest and Ruff,
and offers environment-check, test, and lint tasks plus a diagnostic debugger launch.
It is Windows-specific and intentionally ignored; teammates can select their own
environment interpreter. Recommended extensions are Python, Pylance, Jupyter, and Ruff.

For notebooks, select the repository's `.venv` interpreter. To register a named kernel:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name click2future-ai6132 --display-name "Python (Click2Future)"
```

Use that kernel for project notebooks. Notebook execution uses the same dependencies
as the command-line checks; there is no notebook server or demo service to launch yet.

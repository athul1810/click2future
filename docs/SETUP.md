# Development setup

This setup covers simulation, grouped datasets, evaluation, the untrained denoiser
prototype, CVAE code, and local environment diagnostics. [Model training](MODEL_TRAINING.md)
covers the CVAE pilot and E4 adapter. New diffusion execution work is on B's branch
and awaits integration here; the demo remains to be built. TC2 validation is tracked separately in T02/T03.

## Environment and dependencies

- Python **3.12.15**, pinned in `.python-version`; uv downloads it when needed.
- uv **0.13.0**, required by `pyproject.toml`.
- PyTorch **2.11.0** for both backends: `cpu` or CUDA 12.8 (`cu128`).
- Pymunk, NumPy, Pillow, and PyYAML for numerical/image IO and YAML configurations.
- pytest, Ruff, and ipykernel in the default `dev` dependency group.

`uv.lock` records exact direct/transitive versions, sources, and artifact hashes.
The development group includes pip because editor package views run `python -m pip list`.
If that command reports `No module named pip`, rerun the locked sync command for your
selected extras, then refresh the editor's environment view.
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

## macOS and Linux CPU setup

With an existing Python/pip installation:

```bash
python3 -m pip install --user uv==0.13.0
python3 -m uv sync --locked --extra cpu
.venv/bin/python -m click2future.env_check --device cpu
```

If the distribution forbids user pip installations, install the pinned uv release
using an [official installation method](https://docs.astral.sh/uv/getting-started/installation/),
then use `uv sync --locked --extra cpu`. The locked CPU setup and tests have been checked on macOS;
the contributor also reported Windows checks. Linux and TC2 validation remain pending. Do not change the cluster's system Python,
driver, or CUDA modules. Record its constraints in `COMPUTE.md` before selecting a
compatible training environment and running an explicit CUDA check on TC2.

## Optional image metrics

SSIM and LPIPS require the `image-metrics` extra in addition to the selected backend:

```bash
uv sync --locked --extra cpu --extra image-metrics
```

For a validated CUDA environment, replace `cpu` with `cu128`. Repeat both extras on
future sync/run commands to retain them. TorchVision is pinned alongside Torch for each
backend. [Metric usage](METRICS.md#image-quality) explains the one-time LPIPS weight download,
offline evaluation, and reproduction commands. The default test run skips image-metric
tests if these optional packages are absent; real LPIPS tests also require cached weights.
The full image-metric suite has been checked with the pretrained weights on CPU.

## Verification

```powershell
.\.venv\Scripts\python.exe -m click2future.env_check --device auto --output outputs\environment.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
py -3.12 -m uv pip check --python .\.venv\Scripts\python.exe
git diff --check
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`, use `/`
in output paths, and invoke uv as `python3 -m uv` or `uv`. The installed
`click2future-check` entry point runs the same diagnostic.

The diagnostic reports interpreter/package versions and device details, checks PNG,
NumPy archive, YAML, and NumPy/PyTorch roundtrips, then exercises a convolution's
forward/backward pass, optimizer update, and exact checkpoint reload on CPU and,
when selected, CUDA. An explicit unavailable CUDA request returns exit code 1.
These checks validate the environment, not physics or model quality.

The tests cover CPU reporting, unavailable CUDA, denoiser behavior, and profiling
configuration/failure handling. Add simulator and evaluation tests as those modules arrive.

## Artifacts and configuration

### Profile the untrained architecture

```powershell
.\.venv\Scripts\python.exe -m click2future.profile_pilot --device cuda --output outputs\pilot-profile.json
```

This uses `configs/pilot.yaml` and synthetic tensors. Use `--auxiliary` for the optional
heads, or `--no-auxiliary` to disable configured heads. Model conditioning, object count,
auxiliary heads, and profiling precision are read from YAML unless explicitly overridden
with `--conditioning`, `--max-objects`, `--auxiliary`/`--no-auxiliary`, or `--precision`.
Use `--batch-size`, `--future-frames`, or `--base-channels` to test dimensions.
The report records the resolved settings and checks finite loss and gradients. CPU
profiling supports float32; CUDA supports auto, float32, float16, and supported bfloat16.
Measure on an uncontended GPU; `timing_stable` requires warm-up and at least three
measured steps. Inspect the per-step spread rather
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

Open the repository folder in VS Code and select `.venv/bin/python` on macOS/Linux
or `.venv\Scripts\python.exe` on Windows. The optional `.vscode/` folder is ignored
and is not included in a fresh clone. Configure editor tasks locally if needed.

For notebooks, select the repository's `.venv` interpreter. To register a named kernel:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name click2future-ai6132 --display-name "Python (Click2Future)"
```

Use that kernel for project notebooks. Notebook execution uses the same dependencies
as the command-line checks; there is no notebook server or demo service to launch yet.

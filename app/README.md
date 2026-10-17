# local demo

The browser app runs real local diffusion/CVAE checkpoints. It supports preset
simulator images, click/drag selection, four paired seeds, synchronized playback,
model comparison, cancellation and explicit cached replay. A separate simulator
rollout uses the preset's hidden physical world. Those properties never enter
the neural model.

Checkpoint quality is still experimental. Running the app is not evidence of
held-out action control, calibrated uncertainty or a state-of-the-art result.
Pixel-derived trajectory overlays and a small warm browser timing study are now
implemented. Held-out control and the final presentation rehearsal remain open
in [the specification](DEMO_SPEC.md).

## launch

From the repository root, after the existing environment setup:

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server `
  --dataset outputs/pilot/data `
  --checkpoint "e3 spatial=outputs/pilot/e3/checkpoint.pt" `
  --device cuda
```

Open `http://127.0.0.1:8765`. Checkpoints and datasets are local working artifacts,
not downloaded automatically. Use `--device cpu` on a machine without CUDA.
CPU sampling can be slow; start with one checkpoint. The server only listens on
the local machine. On TC2, inference requires a compute allocation; use the
presentation laptop for the normal interface.

Python sources use LF line endings on Windows and Linux so the CVAE's exact
source-hash check works across both machines. If an older checkout reports a
source mismatch, update the checkout and verify the checkpoint's source revision;
do not bypass its provenance check.

Default presets are the first available example from up to eight validation
scene groups. `--split train` selects development/training examples instead.
For a declared development example, use `--samples sample-id another-id`.
The app permits only train/validation presets when a manifest exists. It refuses
legacy test, ID-test, every OOD/physics-test split and unknown split names before
loading their arrays. Without a dataset manifest, explicit sample
IDs are allowed but labelled as having an unverified split. The operator must
verify those examples before a presentation.

Add repeated `--checkpoint "name=path"` arguments to compare up to four models
on the same image, object, push, horizon and seeds. Supported models use the
current diffusion EMA format or CVAE learned-prior format and must have at
least one training update. Model/config/source incompatibilities fail visibly.
No neural generation is replaced by simulator video or a stored nearest scene.

For the private pilot handoff, every member is checked against its manifest:

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server `
  --bundle outputs/pilot-handoff --models E3-spatial E0-flow-motion --device cuda
```

All six bundled checkpoints load. Choose up to four names from `MANIFEST.json`.
Recorded validation failures and extra-supervision notes appear in run details.
The flow/motion model is an additional supervised variant, not the ordinary CVAE.

## controls and reproducibility

- Click a coloured object, then drag from it. The arrow anchors at its centre;
  positive x is right, positive y is down. Strength is clamped to the common
  declared force limit. Push duration is always 0.25 seconds.
- Reverse/zero buttons and arrow keys change the action. Shift makes a smaller
  keyboard adjustment. Keep the starting seed unchanged for paired comparisons.
- Generate runs four samples with consecutive seeds. Inputs lock until the
  request finishes or is cancelled. Cancellation takes effect between samples;
  an in-progress sample may finish internally but cannot replace the display.
- Every clip includes exact supplied frame 0. Shared playback uses native 8 FPS.
  The simulator panel is a labelled single reference world, not an ensemble or
  the model's prediction.
- Changing inputs labels existing videos as the previous request. Model details
  retain checkpoint/source hashes, device, sampler settings, force and seeds.
  The demo does not certify action adherence or an outcome probability.
- The experimental pixel-track toggle overlays accepted trajectories only.
  Missing observations break paths and remain in the displayed counts. Extractor
  version/settings and failure reasons are in run details. Candidate v2 still
  awaits the independent calibration review; overlays do not certify correctness.

The default cache is `outputs/demo-cache`. Its identity includes observed RGB,
selected mask, force, ordered seeds, horizon, checkpoint, sampling source and
runtime. Cache integrity is checked before replay. A changed hidden reference
world does not change the neural cache key. Replayed outputs are labelled
`cached replay`; the reference is recomputed separately. Use `--cache` with a new
directory for a fresh live rehearsal. Cache files include generated pixels, so
keep them outside the submitted source ZIP.

`--sampling-steps` defaults to 32 DDIM steps; this changes the sampler and cache
identity. Displayed seconds measure model sampling/cache retrieval and pixel
conversion. They exclude reference rendering, PNG transport and browser display;
they are not end-to-end presentation latency statistics.

## verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_demo.py tests/test_submission.py -q
py -3.12 -m uv run --no-project --with playwright python scripts/check_demo_browser.py
```

The optional browser rehearsal uses installed Chrome/Edge in a fresh headless
session. It does not use the personal browser profile. The script writes local
screenshots and an interaction record to `outputs/demo-browser`. It expects a
running 32-frame server with at least two visible objects; two presets also
exercise scene reset. `--seed 432350 --require-live` checks an uncached request;
choose a new seed if that request is already cached. Playwright is a temporary
verification dependency, not an app requirement.

Unit tests check offline sampler parity, private RNG isolation, exact frame 0,
cache identity/integrity, hidden-state separation, cancellation, input validation
and local HTTP boundaries. The browser rehearsal checks 24 interactions with
two presets, plus the pixel-track toggle (25 total). These checks establish
functionality, not model quality.

Warm end-to-end timing, with one warm-up and fresh seeds on every measured request:

```powershell
py -3.12 -m uv run --no-project --with playwright python scripts/benchmark_demo_browser.py --output outputs/demo-timing-new.json
```

This includes sampling, tracking, reference rendering, PNG transfer and the first
painted browser canvas. It rejects cached requests. Startup/checkpoint loading
are excluded; the reported empirical p95 is a small-sample estimate.

## supervised trajectory checkpoint

e5 uses the same observable inputs and public offline sampler. it adds future-pose
supervision during training and a rigid observed-pixel renderer. it is a separate
method, not a drop-in e3 result. see [the measured pilot](../docs/TRAJECTORY_PILOT.md).

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server --dataset outputs/pilot-handoff/presets --checkpoint E5-trajectory=outputs/trajectory-handoff/best.pt --device cuda --trajectory-steps 16
```

or use the new verified e5 bundle with `--bundle` and `--models E5-trajectory`.
`--trajectory-steps` controls edm heun steps; the existing `--sampling-steps`
continues to control pixel ddâ€‹â€‹im sampling. both enter the replay identity.
the checkpoint requires two separately visible objects, native 64px rgb and the
matching model source hash. all current quality limitations remain visible in the
bundle metadata and pixel-track counts.

the [private draft handoff](https://github.com/athul1810/click2future/releases/tag/untagged-bc25eca4153d71c40aec)
contains e5, the unchanged e3 comparison checkpoint, two validation presets and
the complete evidence. use the source from pr #46 or its matching release target.

```powershell
gh release download trajectory-validation-2026-10-17 --repo athul1810/click2future --pattern 'click2future-trajectory-validation.zip*' --dir outputs/trajectory-download
.\.venv\Scripts\python.exe -m zipfile -e outputs/trajectory-download/click2future-trajectory-validation.zip outputs/trajectory-demo-bundle
.\.venv\Scripts\python.exe -m click2future.demo.server --bundle outputs/trajectory-demo-bundle --models E5-trajectory E3-spatial --device cuda
```

archive sha-256: `93d1c7129a82cdc8bf96abdd43dd0fbb2e8eac794d1a2e89a2ef2db72467fd77`.
the uploaded archive was downloaded again and matched byte-for-byte; all 22
inventoried member hashes pass. the bundled launch passes 25 browser checks.
see [handoff checks](../docs/results/trajectory-handoff-checks.json).

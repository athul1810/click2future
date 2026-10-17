# local demo

The browser app runs real local diffusion/CVAE checkpoints. It supports preset
simulator images, click/drag selection, four paired seeds, synchronized playback,
model comparison, cancellation and explicit cached replay. A separate simulator
rollout uses the preset's hidden physical world. Those properties never enter
the neural model.

Checkpoint quality is still experimental. Running the app is not evidence of
held-out action control, calibrated uncertainty or a state-of-the-art result.
Pixel-derived trajectory overlays, the measured latency benchmark and the final
presentation rehearsal remain open in [the specification](DEMO_SPEC.md).

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
two presets. These checks establish functionality, not model quality.

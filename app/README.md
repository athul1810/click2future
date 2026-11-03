# local demo

The browser app runs real local diffusion/CVAE checkpoints. It supports preset
simulator images and native synthetic PNG input, click/drag selection, four paired seeds, synchronized playback,
model comparison, cancellation and explicit cached replay. A separate simulator
rollout uses the preset's hidden physical world. Those properties never enter
the neural model.

Use **save input** to export a preset's exact first image, or **open png** to load
a native 64 x 64 opaque RGB, RGBA or palette PNG. The supported input has a flat
tabletop, uniform border and exactly two separated visible sprites. Files are
limited to 64 KiB. Images are never resized or substituted. Real photographs,
touching/occluded objects and other resolutions are rejected by this input path.
Image-derived masks define the click targets; the model receives only the image,
selected mask and force. All four 32-frame futures preserve the input pixels at
frame zero. Uploaded inputs are marked unscored and have no simulator reference
or invented physical parameters. This input support does not improve the measured
physics quality of the existing checkpoints.

The 2 November interface adds a pinned action comparison. Generate, pin, then
change the push or selected object and generate again. Both complete four-seed
ensembles stay on the same timeline. A different seed list is labelled unpaired;
equal seeds do not identify the same physical world in the CVAE/E5/E6 models.
Changing scenes clears the pin. Each tile also shows its selected-object
displacement in native pixels at the current frame, projected along the recorded
push. Zero-force clips show distance from the start. Missing tracking stays
unavailable; these readouts do not certify physical validity.

The separate reference selector switches among existing hidden-physics variants
with exactly the same first image. It changes only the simulator panel, never
model inputs, generated clips or their cache identity. Reference properties are
displayed in that panel. A dataset containing only one reference has one choice.

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

## combined presentation bundle

The verified local bundle at `outputs/demo-20261102/bundle` contains the ordinary
flow CVAE, E5 and the E6 token checkpoint. It includes the four validation images
and their four hidden-physics references. It is an experimental comparison, not
a promoted physical-quality result. In particular the E6 token checkpoint is not
the later paired-action package and still has passive-motion/contact failures.

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server `
  --bundle outputs/demo-20261102/bundle `
  --models E6-contact E0-flow E5-trajectory --device cuda
```

`scripts/build_demo_bundle.py --recipe local-recipe.json --output new-folder`
builds a new verified bundle from local artifacts. The recipe declares `dataset`,
`samples` and one to four `models`. Each model needs `name`, `checkpoint`,
`checkpoint_sha256`, `quality` and `notes`. The builder checks real model/source
compatibility, preserves checkpoint bytes, records training steps and inventories
all members. Include every matching reference variant in `samples` to enable the
reference-world selector. The output folder must be new.

On the RTX 3060 laptop, all three models together have warm end-to-end median
1.919s and empirical p95 2.232s across ten uncached requests after one warm-up.
This includes tracking, reference rendering, transport and first painted canvas;
it excludes server startup/checkpoint loading and is a small-sample timing result.
The final interface passes 34 real-browser checks. See
[the rehearsal record](../docs/results/demo-rehearsal.json).

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

Each `models` row in a bundle manifest needs `name`, `file`, `checkpoint_sha256`,
`evaluated_checkpoint_step` (the checkpoint's training update), `quality` and
`notes`. Bundle verification rejects a row without them before loading any model.
The published wall bundle below lacks `evaluated_checkpoint_step` and `notes` for
both models, so it stops at that check; add both when building the combined
flow-CVAE/E5/E6 bundle.

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
  version/settings and failure reasons are in run details. Tracker v2 was
  reviewed by Sanjay and merged in #42; overlays do not certify correctness.

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

The later [physics audit](../docs/CONTACT_UNCERTAINTY_AUDIT.md) finds contact
penetration and late zero-force drift despite improved initial action response.
The checkpoint remains experimental; its four futures are not calibrated outcomes.

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server --dataset outputs/pilot-handoff/presets --checkpoint E5-trajectory=outputs/trajectory-handoff/best.pt --device cuda --trajectory-steps 16
```

or use the new verified e5 bundle with `--bundle` and `--models E5-trajectory`.
`--trajectory-steps` controls edm heun steps; the existing `--sampling-steps`
continues to control pixel DDIM sampling. both enter the replay identity.
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

## optional shared-world preview

The 3 November profile adds the existing E7 conditioned checkpoint alongside
E6 token, E0 flow and E5. It samples mass/friction/restitution from the declared
prior and reuses each seed's draw when the push changes. Result metadata retains
all four draws in canonical observed-object order. They are not estimated or
read from the supplied scene. Reference-world switching stays separate.

The profile passed 35 functional browser checks and verified 37 bundle files.
Ten fresh four-model requests after one warm-up took median 3.14 seconds and
empirical p95 3.17 seconds on the RTX 3060 laptop, including the browser display.
Startup and loading are excluded. The earlier three-model measurements remain
in the rehearsal record. These are small local timing samples, not throughput
comparisons with general video models.

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server `
  --bundle outputs/demo-20261103/world-bundle `
  --models E7-prior-preview E6-contact E0-flow E5-trajectory --device cuda
```

E7 still has 11 penetrating clips out of 96 validation samples and fails overall
quality. This optional inspection profile does not promote it. Bundle recipes,
weights and caches remain separate from the source-only submission archive.

## optional one-pass comparison

The separate prior bundle loads `E11-one-pass`, its matched `E11-diffusion`
control, `E0-flow` and `E7-prior-preview`. Both E11 checkpoints are final-update
seed-19 models; the screen reused four validation scenes. One-pass decoding
reduces penetrating clips from 11/96 to 1/96, but adds zero-force drift in 2/16
clips and reduces timed-contact F1. Neither model passes the quality gate.
See [the complete screen](../docs/PRIOR_ONLY_EXPERIMENT.md).

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server `
  --bundle outputs/demo-20261103/prior-bundle `
  --models E11-one-pass E11-diffusion E0-flow E7-prior-preview --device cuda
```

This profile passed 35 browser checks. Ten fresh requests per model, after one
warm-up and with the same input/seed range, took median/p95 0.394/0.402 seconds
for one-pass and 1.286/1.290 seconds for its diffusion control on the RTX 3060
laptop. The four-model profile took 3.286/3.443 seconds. These measurements
include browser display and exclude startup/loading; they measure responsiveness,
not physical quality. The benchmark refuses cache hits:

```powershell
python scripts/benchmark_demo_browser.py --model E11-one-pass --repeats 10 --seed 430300 --output outputs/timing-direct.json
```

Use a fresh output file and restart the server or change the seed range for a
new measurement. Keep the control's seed range matched. The three-seed D018
study is separate and does not change these seed-19 demo checkpoints.

## contact candidate comparison

the 18 october paired-action candidate fixes passive drift on the saved push,
but still fails overall contact/completeness quality. it is an optional comparison,
not a replacement for the frozen handoff. after obtaining the private artifacts:

```powershell
.\.venv\Scripts\python.exe -m click2future.demo.server `
  --dataset outputs/tc2-action-quality/data `
  --samples g03-scene-00730016-forward-p00 g03-scene-00730017-forward-p00 `
  --checkpoint "e6-paired-actions=outputs/tc2-contact-20261018/contact-grid-20261018/best.pt" `
  --checkpoint "e5-frozen=outputs/trajectory-handoff/best.pt" --device cuda
```

the default shows the first model's four futures. select the explicit comparison
option to show both models. native 8 fps playback starts after generation unless
reduced motion is enabled; changing the input pauses the old output. four primary
futures fit common laptop heights. see the [measured comparison](../docs/CONTACT_TRAINING.md)
for the remaining failures; no quality ranking is inferred from the interface.

## matched canvas-loss comparison

The private `wall-training-validation-2026-10-18` draft release contains two
matched E6 checkpoints and all 24 validation presets. Use source `bc77c7d` or
later. Both adapters reproduce four offline-sampler videos exactly on CPU.

```sh
gh release download wall-training-validation-2026-10-18 --repo athul1810/click2future --pattern click2future-wall-validation.zip --dir outputs/wall-download
python -m zipfile -e outputs/wall-download/click2future-wall-validation.zip outputs/wall-demo
python -m click2future.demo.server --bundle outputs/wall-demo --models E6-control E6-wall --device cpu --trajectory-steps 16 --force-limit 1.1 --samples g03-scene-00730016-wall-p00 g03-scene-00730017-wall-p00
```

The explicit force limit includes the native wall action and matches both saved
configs. Keep seeds fixed when comparing. The bundle verifies all 52 inventoried
files before loading. Its ZIP SHA-256 is
`bab8f39a29e87dca13947de4693f497a793997dda608bf14d748693e5f35b6d1`.
Complete clips improve from 88/96 to 95/96; penetration remains in 15/96 and
overall quality still fails. See [the full comparison](../docs/WALL_TRAINING.md).
The bundle is separate from the source-only course ZIP.

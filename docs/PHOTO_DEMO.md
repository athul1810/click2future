# Photo demo

This is a separate real-photo interface. The original 64-pixel models and their
frozen evaluations remain the project's controlled experiment. The photo backend
is the **external pretrained Force Prompting point-force model**, using
CogVideoX-5b-I2V. It is not a model trained by this team, a neural physics repair,
or an extension of the original test results.

## Interaction

Open a JPEG, PNG or WebP photograph. The displayed image is the actual oriented,
RGB, centre-cropped model input. Click a point and drag to aim the push. Arrow
keys adjust the vector; Shift gives smaller increments. Keep the caption and
the four declared seeds fixed when comparing actions. Generate, pin the action,
reverse the push, then generate another set. All samples remain visible with a
shared timeline; expand a video for inspection.

The native model produces 49 frames at 720x480 and 8 fps. Its minimum scalar
still encodes a moving Gaussian condition; physical rest at that setting has
not been established. Strength is the released model's control scale, not
Newtons. Equal noise seeds do not identify the same hidden physical world.
No ground-truth geometry, trajectory or hidden material properties are inferred
or invented by the interface. Four videos are samples, not calibrated confidence.

## Local interface and transport

Use the project's locked environment, adding the `photo-remote` extra if needed.
Choose a new local directory for each backend profile. From the repository root:

```powershell
python -m click2future.demo.photo_server --spool outputs/photo-native --backend force-prompting --port 8770
python scripts/photo_remote.py --host TC2_LOGIN_HOST --username NETWORK_USERNAME --remote /ABSOLUTE/REMOTE/SPOOL --spool outputs/photo-native --backend force-prompting --archive-raw
```

The server prints its loopback address. The transport asks for the password in
the terminal, keeps it only during authentication, and uses the existing SSH
known-hosts file. It does not create or save credentials. Connect through the
university VPN first. Leave the transport process running. Disconnecting it
does not cancel an allocated generation job; reconnect to the same spool.

The 20-step profile is `force-prompting-20`. It has a separate cache identity
and needs a separate spool. PHOTO-P02 selects this profile on one development photo after an exact replay
and inspection of all samples. Do not relabel its
outputs as the published 50-step reference. PHOTO-P01's smaller-resolution
candidate failed at the released model's positional embedding and is not an
available interface option. The measured median is 192.84 seconds per sample versus 466.42 at 50 steps
on that photo. Motion is weak in both. This engineering choice does not establish
a quality gain or reliable force control. The first native 50-step smoke took about 7.8 minutes per sample on A40.

## Allocated worker

The GPU environment is isolated from the original project's Python/Torch
environment. The prepared environment uses Python 3.10, Torch 2.5.0/cu121,
Diffusers 0.33.1 and the official custom Force Prompting pipeline. The project
worker imports its small queue contract from source; do not install the main
project's Torch 2.11 environment into this backend.

Start from a committed, immutable uploaded project. Set `C2F_PHOTO_PROJECT`,
`C2F_PHOTO_NODE_ROOT` and `C2F_PHOTO_SPOOL`, then submit
`cluster/photo_worker.sbatch` on the node holding the verified weights.
`C2F_PHOTO_BACKEND` selects the same profile as both local processes. The
worker requires `repo/ARTIFACTS.json` with a passed official-artifact check.
`cluster/verify_photo_artifacts.py` creates that manifest after checking all
selected model files against their pinned official repository identities and
the external source inventory. Keep the source inventories and model licences.

The external source revision is `ef3d0f847f787552b5ce004906c4944292e58996`.
The base revision is `a6f0f4858a8395e7429d82493864ce92bf73af11`; the point-force
checkpoint release is `bc1a3faa381a8c935ebce980a18afb597d4a15c4`.
The runtime loads every transformer and ControlNet layer. It does not enable
the upstream layer-truncation hack. Manual captions avoid an external caption
API. Large weights, environments and raw arrays use node-local storage because
home-directory quota is independent of filesystem free space.

## Records and presentation use

Each immutable request binds input pixels, crop, caption, point/vector, four
seeds, checkpoint/runtime identity and video contract. Completed output needs
four matching, hash-checked MP4s. Failed/cancelled attempts remain on disk.
After each finished sample, the worker publishes a separate incomplete attempt
manifest. The transport mirrors those videos/raw arrays even if a later sample
fails or the request is cancelled. It never treats partial output as four ready
futures. A hard allocation failure can still interrupt a transfer, so preserve
the node workspace and recover missing artifacts before releasing that node.
The transport backs up raw arrays through the existing Slurm allocation when
`--archive-raw` is enabled. Finish this backup before the allocation expires;
node-local storage is not durable. Back up the complete local spool separately.

An exact prior request returns **cached replay**, including while the worker
is offline. New requests require a ready worker. Cached output, live output,
generation failure and unavailable transport are separate states. Retain and
show all four declared samples; do not select the best-looking seed.
The recorded-run selector restores the saved image, caption, point, vector and
four samples without pretending to submit fresh inference. The latest complete
recording opens on startup. Reopening a recording for the same photograph keeps
the pinned comparison. Integrity is checked when the recording is opened.

The original simulator/CVAE/diffusion comparison and this external photo
extension need separate labels in slides and recordings. PHOTO-P01/P02 are
engineering development studies, not untouched evaluation or SOTA evidence.
The photo panel and rating rules must be frozen before any claimed evaluation.

## StreamForce candidate

The separate `cluster/streamforce_worker.py` remains gated by a completed native
four-seed probe plus the seed-0 repeat. It checks the probe's native frame
contract, source/artifact/loader identities, library versions and compiled
attention binary before accepting work. The initial model-structure and small
attention-kernel checks alone do not enable it.

Each action loads the model in a separate GPU process and generates its four
seeds there. Cancellation terminates that process before the parent publishes
the final state. Finished samples retain their incomplete-attempt records.
The next action starts with a fresh process, including text encoding. This
avoids reusing an interrupted asynchronous decoder, at the cost of a cold load
for each action. End-to-end timing and native cancellation still need to pass
before this candidate replaces the existing backend.

Use a new `streamforce` spool. Set `C2F_PHOTO_PROJECT`, `C2F_STREAMFORCE_ROOT`,
`C2F_STREAMFORCE_PROBE` and `C2F_PHOTO_SPOOL`, then submit
`cluster/streamforce_worker.sbatch` on the node holding the verified files.
The candidate's native contract is 81 frames, 832x480 at 16 fps with four
denoising steps. This describes its interface, not a measured A40 speed result.

The completed native smoke and recorded-browser checks are recorded in
[the evidence summary](results/photo-native-probe.json). Its official caption
already requests swinging; these four clips are not a force-only control test.


## Paired native run and offline fallback

PHOTO-P03 completes both opposite pushes through the actual browser, transport
and A40 worker. Each retains seeds 0-3. The two four-sample requests take 788.88
and 792.06 seconds through the browser. All eight raw RGB/video hashes match
the local backup. Right-push seeds 0/1 respond weakly; every reversed-push sample
translates the mug left. These are manually inspected outputs on one public
StreamForce development image, without physical ground truth or calibration.

PHOTO-OPS01 checks cancellation before any completed sample and after one
finished sample has been backed up. An incomplete request never becomes four
ready futures. The prior result stays visible. The interrupted transport is
also recovered under the original request IDs without resampling.

Keep a separate offline spool snapshot with its actual recorded model identity.
Start the same server against that snapshot with `--backend force-prompting-20`.
It restores saved requests, replays matching actions, and refuses unseen
requests while disconnected. Never attach the live transport to that frozen
snapshot. Dependency installation is still required before going offline.

The paired-run evidence and limitations are in
[the integration summary](results/photo-paired-integration.json). The original
models' physical quality and final presentation-machine rehearsal remain open.

# Compute and feasibility

**Training resource: the course-provided TC2 GPU Cluster.** CCDS has made TC2 available to MSAI, MSDS, and MSCS students for course-related work. We used TC2 for model training and the photograph inference workers.

## Access and account limits

TC2 access is approved for **BOBAN ATHUL KRISHNA** (`athulkri001`) and **MAITI SHRIJEET** (`shrijeet001`), according to their access emails. Shrijeet supplied approval on 16 October 2026. Access for the remaining members has not been recorded.

The email lists these Slurm settings:

| Item | Confirmed value |
|---|---|
| Partition | `MGPU-TC2` |
| QoS | `normal` |
| Per-user resource limits (`MaxTRESPU`) | 10 CPUs, 1 GPU, `30G` system memory |
| Running-job limit per user (`MaxJobsPU`) | 2 |
| Observed submission limit | A third queued/running job was rejected with `QOSMaxSubmitJobPerUserLimit` on Athul's account; submit the next dependency after one finishes |
| Maximum job duration (`MaxWall`) | 6 hours (`06:00:00`) |
| GPU model and VRAM | Job 42886 on TC2N01: NVIDIA A40; 46,068 MiB reported by nvidia-smi; 44.43 GiB visible to PyTorch |
| Total GPU-hour quota and access expiry | Not stated in the email |
| CUDA, driver, Python, framework | CUDA runtime 12.8; driver 570.148.08; Python 3.12.15; PyTorch 2.11.0+cu128; glibc 2.34 |
| Dataset/checkpoint storage quota | No personal quota returned by the recorded check |

The resource limits apply across the user's running jobs. Two job slots do not provide two GPUs. The `30G` limit is system RAM, not GPU VRAM. Plan single-GPU training and save resumable checkpoints before the six-hour limit, including model, optimizer, scheduler, step, and random-number states.

## Remaining access checks

- [x] Record access approval and the account limits.
- [x] Verify Shrijeet's VPN route and head-node TCP connectivity (16 October 2026).
- [x] Complete the first interactive login (reported by Shrijeet, 16 October 2026); the email says this creates the home directory.
- [x] Run a small scheduled GPU job and record GPU model, VRAM and software versions: job 42886, completed with exit 0:0 on 16 October 2026.
- [ ] Check storage limits, any total usage quota, and access expiry.
- [x] Keep code/configs in Git and full private checkpoint/result bundles under ignored local `outputs/tc2-ops`, copied from TC2 after each completed pilot. TC2 provides no backup service; datasets and model files stay out of Git and the course submission.

Athul's account independently completed setup job 43158 on TC2N01 with an A40
and the locked CUDA/evaluation environment. `mytcinfo` confirms the same normal
QoS limits. `quota -s` returned no personal quota, and the checked account metadata
does not expose total GPU-hours or service expiry. These values remain unknown;
they do not indicate a pending login or GPU setup. Default setup job 43177 also passed the native CVAE CUDA backward check and
360 tests plus four subtests; four LPIPS-weight tests skipped because TC2 lacks
those cached weights. All 364 pass locally with cached weights. T02 is complete. Final model
quality remains part of T09/T10, and quota information should be used if support
later provides it. Trained CVAE checkpoints and GPU-generated samples from jobs
43165/43166 are backed up under ignored `outputs/tc2-ops/cvae-pilot/collected`.

Shrijeet's initial SSH attempt closed, followed by a successful plain interactive login
and confirmed home-directory creation on 16 October. Setup job `42886` completed on
`TC2N01` in 13 minutes 28 seconds, using source revision `47ca5e0`. The environment,
70-package compatibility check, CUDA forward/backward and weight reload, 147 tests
and four subtests passed. One optional image-metric module was skipped because its
extra was not installed. Exact measurements, source/export hashes and scope limits
are in the [allocation evidence](results/tc2_probe_2026-10-16.json).

The 64px, 31-future-frame, batch-one denoiser profile used bfloat16 and averaged
0.0380 seconds per synthetic forward/backward/optimizer step after three warm-ups
(five measured steps). Peak allocated GPU memory was 665.39 MiB; peak reserved memory
was 798 MiB. This is a short synthetic profile, not end-to-end training or sampling
latency. The real-data measurement and full-state resume appear below. A chosen
full dataset still needs its own throughput and quality gate before main runs.

The email says accounts that never attempt
login may be revoked after three months. Use TC2 for the approved coursework. Keep
credentials outside the repository. See the [TC2 workflow](../cluster/README.md) for
the prepared upload, inventory, installation, and probe scripts. The validated source
release is `~/click2future/releases/47ca5e0d9e9246d929751c00d15e8bec2d1f7e68/`.
Other nodes or later allocations may differ; check their device reports.

Track the login and GPU validation results in T02. CPU simulation, data preparation, and development checks can proceed independently.

## Real data pilot, 16 October

Jobs 42909 and 42910 completed on TC2N01 with exit code 0:0. Step 201 matched
the uninterrupted update across allocations with zero tensor difference in model,
EMA, optimizer, scheduler, random state and data order. This verifies complete
single-device continuation on that node, rather than merely reloading weights.

Four training clips at 64px/31 future frames, width 32, batch one and bfloat16
measured 0.04883 s/step including loading, excluding the first ten measured steps.
Peak allocated memory was 752,562,176 bytes. Four sequential 32-step DDIM samples
took 1.744 s after earlier GPU work; first-use latency is a separate measurement.
The model failed the video-quality gate. Both EMA and raw weights failed tracking
with 32/128/1000 sampler steps. Job 42917 completed a 10,000-step independent
restart without passing video quality. It measured 0.04657 s/step; the allocation
took 10 minutes 1 second including startup and sampler diagnostics. Job 42920
tested velocity prediction for 2,000 steps in 1 minute 59 seconds; it reduced
visible noise but recovered only 1-7/62 future object observations. See
[follow-up evidence](results/diffusion_debug_2026-10-16.json).

The full checkpoint/result bundle was copied to ignored local storage outside
TC2. User storage quota and access expiry remain unconfirmed; `quota -s` exposed
no user-specific limit. Shared-filesystem capacity is not a personal quota.
See [pilot evidence](results/diffusion_pilot_2026-10-16.json). Do not extrapolate
four-clip cache timings into a final suite budget before measuring the chosen data.

## Measured main-comparison cost

The final E0/E2/E3 records supersede the original planning worksheet.
Values below come from `docs/results/final-cost.json`.

| Model | Training budget | Training time per run | Sampling time per video |
|---|---|---|---|
| E0 flow CVAE | 20,000 updates, batch two | 742-757 s | 0.012-0.015 s |
| E2/E3, original budget | 5,000 updates, batch one | 246-281 s | about 0.54 s |
| E2/E3, matched clip count | 40,000 updates, batch one | 1,965-1,970 s on A40 | 0.58-0.59 s |

E0 seeds 1/2 and the original diffusion runs used A40 GPUs; E0 seed 0's GPU
type was not recorded. The longer diffusion seed-0 pair used A40s; seeds 1/2
used L40S GPUs and took 1,130-1,156 seconds. Sampling was measured on A40 and
includes checkpoint loading and video export. These are not browser timings.

E0 has 574,226 parameters and E2/E3 have 964,035. The longer diffusion budget
matches E0's number of clips seen, not optimizer updates or GPU time. Better
image quality at that budget did not resolve object identity or action control.

The submitted experiments remain at 64px and 32 total frames. No 128px or
latent-autoencoder result is claimed. The setup and four-clip measurements
above are dated pilot records, not estimates of final model quality.

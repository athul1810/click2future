# Compute and feasibility

**Training resource: the course-provided TC2 GPU Cluster.** CCDS has made TC2 available to MSAI, MSDS, and MSCS students for course-related work. The team will use this resource for training.

## Access and account limits

TC2 access is approved for **BOBAN ATHUL KRISHNA** (`athulkri001`) and **MAITI SHRIJEET** (`shrijeet001`), according to their access emails. Shrijeet supplied approval on 16 October 2026. Access for the remaining members has not been recorded.

The email lists these Slurm settings:

| Item | Confirmed value |
|---|---|
| Partition | `MGPU-TC2` |
| QoS | `normal` |
| Per-user resource limits (`MaxTRESPU`) | 10 CPUs, 1 GPU, `30G` system memory |
| Running-job limit per user (`MaxJobsPU`) | 2 |
| Maximum job duration (`MaxWall`) | 6 hours (`06:00:00`) |
| GPU model and VRAM | Job 42886 on TC2N01: NVIDIA A40; 46,068 MiB reported by nvidia-smi; 44.43 GiB visible to PyTorch |
| Total GPU-hour quota and access expiry | Not stated in the email |
| CUDA, driver, Python, framework | CUDA runtime 12.8; driver 570.148.08; Python 3.12.15; PyTorch 2.11.0+cu128; glibc 2.34 |
| Dataset/checkpoint storage quota | To record |

The resource limits apply across the user's running jobs. Two job slots do not provide two GPUs. The `30G` limit is system RAM, not GPU VRAM. Plan single-GPU training and save resumable checkpoints before the six-hour limit, including model, optimizer, scheduler, step, and random-number states.

## Remaining access checks

- [x] Record access approval and the account limits.
- [x] Verify Shrijeet's VPN route and head-node TCP connectivity (16 October 2026).
- [x] Complete the first interactive login (reported by Shrijeet, 16 October 2026); the email says this creates the home directory.
- [x] Run a small scheduled GPU job and record GPU model, VRAM and software versions: job 42886, completed with exit 0:0 on 16 October 2026.
- [ ] Check storage limits, any total usage quota, and access expiry.
- [ ] Choose backup locations for code, configs, run records, and checkpoints. TC2 provides no backup service; keep datasets and model files out of Git and the course submission.

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
latency. A complete trainer/data-pipeline measurement and full-state resume across
jobs remain required before budgeting long experiments.

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
with 32/128/1000 sampler steps. Job 42917 tests a 10,000-step independent restart;
at the measured warm rate its training portion projects to roughly 8.1 minutes,
excluding environment startup, diagnostics and queue time.

The full checkpoint/result bundle was copied to ignored local storage outside
TC2. User storage quota and access expiry remain unconfirmed; `quota -s` exposed
no user-specific limit. Shared-filesystem capacity is not a personal quota.
See [pilot evidence](results/diffusion_pilot_2026-10-16.json). Do not extrapolate
four-clip cache timings into a final suite budget before measuring the chosen data.

## Profiling sequence

1. Run a device/memory smoke check on TC2 and record the environment.
2. Generate a small simulator dataset on CPU; measure samples/second and bytes/clip.
3. For the proposed pixel pilot, profile the denoiser directly. If a latent path is
   selected, first gate its autoencoder on object-boundary/contact preservation.
4. Overfit a tiny fixed video batch, then independently check held-out action response.
5. Profile the compact 64px, 32-total-frame model (31 predicted frames): peak VRAM, seconds/step, throughput, checkpoint size, and four-sample latency. The synthetic batch-one profile is complete; dataset training and sampling remain unmeasured.
6. Calculate the total ablation budget, including evaluation and failed-run reserve, before launching long jobs.

Measure after warm-up; state precision, batch size, gradient accumulation, and hardware. A successful tiny-batch overfit check establishes basic plumbing, not generalization.

## Budget worksheet

```text
training GPU-hours per run = measured seconds/step × planned steps × GPU count / 3600
suite GPU-hours = sum(per-run cost × seed count) + autoencoder cost + evaluation cost
planned allocation = suite GPU-hours + explicit retry/debug reserve
raw RGB bytes = clips × frames × height × width × 3
```

Queue time and CPU data generation affect wall time even when GPU-hours fit. Measure masks, oracle arrays, and compression separately. Keep at least one recoverable checkpoint per active run and use resumable jobs when time limits require them.

| Component | Steps / seeds | Measured time | Estimated GPU-hours | Decision |
|---|---|---|---|---|
| Autoencoder, if latent path chosen | TBD | Pending | Pending | Reconstruction gate |
| E0 CVAE | TBD | Pending | Pending | Profile |
| E1 first-frame diffusion | TBD | Pending | Pending | Profile |
| E2 numeric-action diffusion | TBD | Pending | Pending | Profile |
| E3 spatial conditioning | TBD | Pending | Pending | Profile |
| E4 auxiliary supervision | TBD | Pending | Pending | Profile |
| Evaluation / sampling | TBD | Pending | Pending | Reserve |

## 128px gate

Proceed only if the 64px baseline works, the core comparisons fit the remaining allocation, and a measured 128px profile fits memory and the deadline. Doubling both image dimensions quadruples raw pixels; actual training cost depends on the architecture and must be measured. Preserve scene identities and split membership when rerendering at 128px.

## Fallback order

1. Keep the final resolution at 64px.
2. Reduce clip length, channel width, batch size, and latent size if applicable as measured needs dictate.
3. Reduce dataset/run size coherently across comparisons; report the resulting limits.
4. Use fewer seeds for exploratory runs and reserve repeats for the main claims.
5. Drop optional FVD, additional supervision ablations, and friction-conditioned generation.

Do not assume a particular large GPU or promise a fixed training duration before profiling.

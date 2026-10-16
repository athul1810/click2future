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
| GPU model and VRAM | To check in an allocated GPU job |
| Total GPU-hour quota and access expiry | Not stated in the email |
| CUDA, driver, Python, framework | To record after the device check |
| Dataset/checkpoint storage quota | To record |

The resource limits apply across the user's running jobs. Two job slots do not provide two GPUs. The `30G` limit is system RAM, not GPU VRAM. Plan single-GPU training and save resumable checkpoints before the six-hour limit, including model, optimizer, scheduler, step, and random-number states.

## Remaining access checks

- [x] Record access approval and the account limits.
- [x] Verify Shrijeet's VPN route and head-node TCP connectivity (16 October 2026).
- [ ] Attempt the first login using the supplied user guide; the email says this creates the home directory.
- [ ] Run a small scheduled GPU job using the supplied sample script, then record GPU model, VRAM, and software versions.
- [ ] Check storage limits, any total usage quota, and access expiry.
- [ ] Choose backup locations for code, configs, run records, and checkpoints. TC2 provides no backup service; keep datasets and model files out of Git and the course submission.

Shrijeet's first SSH attempt did not authenticate; account activation through login and
the allocated GPU test are still pending. The email says accounts that never attempt
login may be revoked after three months. Use TC2 for the approved coursework. Keep
credentials outside the repository. See the [TC2 workflow](../cluster/README.md) for
the prepared upload, inventory, installation, and probe scripts. Linux installation
and allocated GPU execution still need live verification.

Track the login and GPU validation results in T02. CPU simulation, data preparation, and development checks can proceed independently.

## Profiling sequence

1. Run a device/memory smoke check on TC2 and record the environment.
2. Generate a small simulator dataset on CPU; measure samples/second and bytes/clip.
3. For the proposed pixel pilot, profile the denoiser directly. If a latent path is
   selected, first gate its autoencoder on object-boundary/contact preservation.
4. Overfit a tiny fixed video batch, then independently check held-out action response.
5. Profile a compact 64px, 16-frame model: peak VRAM, seconds/step, throughput, checkpoint size, and four-sample latency.
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

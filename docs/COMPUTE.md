# Compute and feasibility

**Allocation status: pending confirmation.** The instructor directed the team to the TC2 GPU Cluster coursework application on the course site. GPU type, VRAM, hours, queue policy, and storage are unknown.

## Record before scheduling training

| Item | Confirmed value |
|---|---|
| GPU model / count / VRAM | Pending |
| Allocation and expiry | Pending |
| Maximum job duration / concurrent jobs | Pending |
| CUDA, driver, Python, framework | Pending |
| Dataset / checkpoint storage and quota | Pending |
| Access procedure and scheduler | Pending; keep credentials outside Git |

## Profiling sequence

1. Confirm the environment and run a device/memory smoke check.
2. Generate a small simulator dataset on CPU; measure samples/second and bytes/clip.
3. Train the frame autoencoder on a small subset and inspect reconstruction of object boundaries.
4. Overfit a tiny fixed video batch to check the model and loss plumbing.
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
| Autoencoder | TBD | Pending | Pending | Profile |
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
2. Reduce clip length, latent size, channel width, and batch size as measured needs dictate.
3. Reduce dataset/run size coherently across comparisons; report the resulting limits.
4. Use fewer seeds for exploratory runs and reserve repeats for the main claims.
5. Drop optional FVD, additional supervision ablations, and friction-conditioned generation.

Do not assume a particular large GPU or promise a fixed training duration before profiling.

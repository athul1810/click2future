# Compute and feasibility

**Training resource: the course-provided TC2 GPU Cluster.** CCDS has made TC2 available to MSAI, MSDS, and MSCS students for course-related work. The team will use this resource for training.

## Application and access

Complete the application form attached to the course announcement and email it to [CCDS GPU Support](mailto:ccdsgpu-tc@ntu.edu.sg).

- [ ] Obtain the application form attached to the course announcement and read its instructions.
- [ ] Complete and send the form to `ccdsgpu-tc@ntu.edu.sg`.
- [ ] Record the submission and access status in T02; keep personal application details and credentials outside the repository.
- [ ] Follow the access instructions received from support and verify a small GPU job.

Track application and access status in T02.

Follow the [TC2 access and diagnostic workflow](../cluster/README.md) and
[application template](TC2_ACCESS.md). Inventory, isolated installation, and GPU probe
jobs are prepared; actual account approval, network access, Linux compatibility and
allocated GPU validation remain pending. No TC2 job has been submitted by this setup.

Local CPU work can cover simulation, data preparation, and development checks while TC2 access is arranged. Size training experiments to the TC2 capacity and quota assigned to the team.

## Record before scheduling training

| Item | Confirmed value |
|---|---|
| Training resource | TC2 GPU Cluster, confirmed course resource |
| GPU model / count / VRAM | To record from the TC2 assignment |
| Usage quota and expiry | To record |
| Maximum job duration / concurrent jobs | To record |
| CUDA, driver, Python, framework | To record |
| Dataset / checkpoint storage and quota | To record |
| Access procedure and scheduler | To document; keep credentials outside Git |

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

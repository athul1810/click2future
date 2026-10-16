# Pixel diffusion pilot

Adopted scope: 64px, 32 total frames at 8 FPS, two objects first, then 2-4.
The model generates 31 frames jointly. Frame 0 is retained exactly. The fixed
force lasts 0.25 s; divide its physical vector by 0.5, with no silent clipping.
Only frame-0 RGB, selected mask and force enter the inference boundary.

## Commands

Generate data with `configs/simulator/pilot.toml` and the existing generation
CLI. `configs/training/pilot_diffusion.json` intentionally selects only four
training clips for a diagnostic. Copy it and set `limit_train` to null for an
actual dataset run. Freeze the run's configuration before starting it.

```powershell
python -m click2future.training.train_diffusion --dataset outputs/pilot-data --run-dir outputs/pilot-run --config configs/training/pilot_diffusion.json --device cuda --until 200
python -m click2future.training.train_diffusion --dataset outputs/pilot-data --run-dir outputs/pilot-run --config configs/training/pilot_diffusion.json --device cuda --resume outputs/pilot-run/checkpoint.pt
```

On TC2 submit `cluster/train_pilot.sbatch` with `first`, then `resume` through
an `afterok` job dependency. Set `C2F_PROJECT`, `C2F_PILOT_ROOT` and
`C2F_SOURCE_REVISION`. All application execution stays inside Slurm.
This two-job diagnostic generates separate local data, trains to step 200,
compares the next update across allocations, then continues to step 2000 and
exports paired forward/reverse/zero samples. It does not overwrite dataset-0.1.

## Training and sampling contract

- Cosine DDPM schedule, 1000 timesteps, uniform clip timestep, epsilon MSE.
- AdamW, learning rate 1e-4 with cosine decay to 1e-5 at the immutable horizon,
  gradient norm clipped to 1, EMA decay 0.995. These are pilot defaults.
- Float32 or CUDA bfloat16; neither uses a gradient scaler. No mixed float16.
- E1/E2/E3 share the same backbone. E4 adds the separately reviewed low-noise
  canonical trajectory/contact losses. Fixed collision positive weight must
  come from training labels only, capped at 20; default is 1.
- DDIM eta=0 uses a private generator per sample, endpoint timesteps included,
  x0 clipped to [-1,1], epsilon recomputed after clipping. Test 16/32/64 steps
  on validation before freezing sampling cost. No classifier-free guidance.
- Checkpoints save model, EMA, optimizer, LR scheduler, step, Python/NumPy/
  Torch/CUDA RNG, data permutation/cursor and permutation generator. Resume
  rejects changed config, actual training-file hashes, device type or Torch
  version. CPU continuation is tested for exact equality. Cross-node exactness
  additionally depends on deterministic kernels and hardware/software matching.
- The manifest and split hashes are verified, scene-group membership checked,
  and actual selected training files hashed. No test arrays are used to fit or
  select a model. The fixed training-noise probe is a plumbing diagnostic.

Prepare a condition NPZ with `first_frame: uint8[H,W,3]`,
`selected_mask: bool[H,W]`, and `force_xy: float32[2]` in physical units:

```powershell
python -m click2future.training.sample_diffusion --checkpoint outputs/pilot-run/checkpoint.pt --condition outputs/condition.npz --output outputs/videos.npz --device cuda --seeds 100 101 102 103 --sampling-steps 32
```

The output key `videos` is uint8 `[K,T,H,W,3]`, directly accepted by the video
evaluator. The GIF is for inspection. The JSON records checkpoint/input hashes,
EMA selection, seed policy and latency. On TC2 add `--require-slurm`.

## Gates before scaling

1. Numerical equations, seed isolation, oracle-field exclusion and complete
   interrupted training pass their tests.
2. The same next update matches across two allocated jobs. Keep a recoverable
   checkpoint and copy important experiment artifacts off TC2.
3. Tiny training scenes retain objects, produce coherent motion and reconstruct
   their futures from sampled noise. Inspect videos and extracted tracks;
   declining noise MSE alone does not pass this gate.
4. Independently test held-out zero/reversed pushes and selected-object changes,
   with common seeds and fixed contact-free windows. Report tracking failure.
5. Set a dataset/run budget from measured end-to-end step and four-sample times.
   Start matched E2/E3 runs; main claims need repeats and scene-group intervals.

CVAE and E2/E3 are the core comparison. E4 is retained only if generated-pixel
metrics improve; head accuracy alone cannot establish that. Latents, 128px,
goal search and external-method adaptations receive time only after these gates.

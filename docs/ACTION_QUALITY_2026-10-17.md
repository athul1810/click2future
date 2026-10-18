# action quality follow-up

The original multi-scene RGB models do not generate usable held-out tracks. The
flow variants preserve appearance but fail reversal and changed-selection controls.
This follow-up targets that failure. It does not change the accepted scoring
thresholds, adopt candidate tracker v2 without review, or establish a SOTA result.

## completed integration

- Downloaded the six-checkpoint private handoff and verified archive SHA-256
  `97cc2648128b2b24290059075737a9d285d5c4d3d2f35a2869b15fffa20c0b20`.
  Every member hash and all six CPU checkpoint loads match the manifest.
- `--bundle` verifies the inventory before loading checkpoints or preset arrays.
  It carries the recorded validation failures and supervision notes into run details.
- The demo compares E3 with the motion-supervised flow CVAE on two validation
  presets. Pixel paths use the shared candidate extractor, with missing observations
  retained as gaps and counts. These are generated-pixel tracks, not auxiliary heads.
- The browser rehearsal passes 25 checks. Five fresh four-future requests per model,
  after one warm-up, give a 4.69s median and 4.74s empirical p95 through the first
  painted canvas on an RTX 3060 laptop. Startup/loading are excluded. This small
  sample estimates latency, not a population bound or model quality.

The exact timing record is [demo-pilot-latency.json](results/demo-pilot-latency.json).
The pixel rescore in [diffusion-tracker-v2-checks.json](results/diffusion-tracker-v2-checks.json)
reproduces 247/248 RGB and 194/248 residual observations from unchanged #39/#41
videos. Both original and corrected sampler artifacts are included with archive
and pixel hashes. Historical v1 reports remain unchanged; source-text hashes use
LF normalization so the replay specification works across Windows and Linux.

## current external methods

Primary sources checked 17 October; these are reported capabilities, not reproduced
scores on our dataset. There is no common leaderboard establishing a universal winner.

| method | released evidence | relevance and limits |
|---|---|---|
| [StreamForce](https://neu-vi.github.io/StreamForce/) / [code](https://github.com/neu-vi/StreamForce) | SIGGRAPH Asia 2026 project, causal local/global force control, released code and linked weights. The repository specifies a Wan2.2 5B base, roughly 40 GB for inference without captioning, Hopper fast paths and eight-GPU training. | A strong current external control system. Its 480x832/16 FPS streaming interface and force representation differ from our 64px/8 FPS fixed push. A40 fallback speed and matched force calibration are unmeasured. A reproduction needs its own pinned environment and allocation. |
| [PhysCtrl](https://cwchenwang.github.io/physctrl/) / [code](https://github.com/cwchenwang/physctrl) | NeurIPS 2025; diffusion generates 3D point trajectories from force and physical parameters, then guides video rendering. | Supports separating motion from appearance. Its physical-parameter inputs are unavailable in our task; copying that interface would change the information budget. |
| [PlaySlot](https://play-slot.github.io/PlaySlot/) / [code](https://github.com/angelvillar96/PlaySlot) | ICML 2025; object representations and latent actions support stochastic prediction and planning. | Supports object-level dynamics and structured stochastic futures. Latent actions are not our calibrated physical forces. |
| [DIAMOND](https://diamond-wm.github.io/) | Released NeurIPS 2024 diffusion world model with a small pixel backbone. | Useful released baseline direction and preconditioning reference; Atari action/history interfaces require an explicit adaptation. |

Downloading a large pretrained generator alone does not establish a stronger project.
Our immediate experimental question is whether structured motion can pass the
controls that the current pixel and dense-flow models fail.

## next model and gates

Implemented separate supervised variant: infer two object regions from frame-0 RGB
and the selected mask, condition a compact trajectory diffusion model on their
visible geometry and the routed force, then render sampled rigid motions using the
observed object pixels. Hidden mass/friction/restitution remain unavailable.
Future poses supervise training only. No simulator supplies model futures.

The motion representation uses centre displacement and relative rotation rather
than predicting a complete RGB movie. The selected object's action is explicit in
its token. A shared interaction model handles both objects; square-world symmetry
augmentation transforms images, forces and training targets together. Contact
propagation and hidden-physics uncertainty still have to be learned and measured.

1. Calibrate frame-0 region extraction and an oracle-motion renderer ceiling on
   training/validation only. Report failures and rasterization bias. Oracle-motion
   renders are diagnostic controls, never learned predictions.
2. Pass CPU numerical/gradient, coordinate, private-seed and observable-boundary
   tests. Checkpoint state includes source, data, normalization and sampler settings.
3. Run one bounded TC2 pilot on the frozen 336 training/84 validation clips. Use
   validation only for selection; reserve all five final-test splits. Record new
   compute separately from Athul's completed 82-minute pilot.
4. Evaluate generated RGB with the same tracker, trajectory/action/selection tests
   and static-copy control. Report generated trajectory-head numbers separately.
   Appearance preservation alone cannot pass the action gate.
5. Expand only after useful held-out action response. A matched external adaptation,
   independent training seeds, calibration and OOD evaluation precede superiority
   claims. Goal-region planning remains gated on usable sampled motion.

This changes model representation and uses pose supervision. It must be named as
an additional method, not silently substituted for E2/E3 or called a pure spatial
conditioning ablation. Ananya retains CVAE ownership and shares evaluation with Athul.

the first trained e5 pilot and its remaining contact failures are recorded in
[trajectory_pilot.md](TRAJECTORY_PILOT.md). the original model results and thresholds
stay unchanged.

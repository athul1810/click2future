# Click2Future demo specification

Planning specification, 16 October 2026. The app and trained checkpoints do not exist yet.
The confirmed PPT date is 14 November. Read the
[execution plan](../docs/EXECUTION_PLAN_2026-10-16.md) for delivery gates and pending ownership.

## One experience to deliver

Select an object, specify a push, and see four coherent learned futures. Change the
push and see the consequences change. Compare the learned methods and reveal why
hidden physical properties permit several outcomes from the same visible condition.

The strongest visual moment is propagation: the pushed object contacts another object
and changes its motion. Four noisy or almost-static videos are not sufficient.

## Screen and interaction

| Area | User sees / does |
|---|---|
| Input canvas | Large crisp scene; click highlights an object; drag draws a force arrow anchored at its centre |
| Main output | Four synchronized video tiles, common playback timeline and seed controls |
| Model comparison | CVAE, numeric diffusion, spatial diffusion, and physics-supervised model if implemented |
| Reference panel | Clearly labelled simulator rollout or reference ensemble; optional hidden-property reveal |
| Evidence panel | Pixel-derived tracks/endpoints, useful counts and measured latency; advanced provenance collapsible |
| State feedback | Loading/progress, cancellation, invalid selection, failure, and live/cached replay status |

Use a restrained, high-contrast layout, consistent object colours, readable typography
and spacious video tiles. Enlarge the original 64px output cleanly. Upscaling for display
does not change the generation resolution or evaluation inputs. Do not add a second
learned upscaler before the core works.

Use an optional endpoint-density overlay and a shared timeline to make uncertainty and
contact propagation legible. An educational method view can show a few actual denoising
states captured by the sampler; use the same final output and measure capture overhead.
This view is a small presentation enhancement, not a substitute for good generated motion.

Keep the UI in one lightweight local application with a browser canvas and Python inference
entry point. Choose the simplest implementation that supports actual click/drag and video
playback; a separate frontend build system is not a prerequisite. Shrijeet owns demo
integration. Ananya's proposed CVAE lead role is awaiting acceptance.

## Exact behaviour

- Selection uses the preset's frame-0 mask. A click selects an object; it does not set
  an off-centre force or torque. The arrow snaps to the object's centre.
- Map display coordinates back to the 64px scene. Positive x is right and positive y
  is down. Clamp to the declared force range and display the actual applied strength.
- Force duration is fixed at 0.25 seconds. Mouse drag duration is not physical duration.
- Use the same normalization, condition construction and sampler as offline evaluation.
  A request includes the actual frame, selected mask, force and ordered sampling seeds.
- Return all 32 frames at 8 FPS, including exact supplied frame 0. Score only future frames.
- Preset/reference state and neural conditioning are separate. Hidden mass, friction,
  restitution, future masks and simulator trajectories cannot enter neural inference.
- Revealing/changing hidden reference physics leaves a fixed-seed model distribution
  unchanged when the visible condition is unchanged. A known-physics model would be a
  separate model/input setting and must be labelled accordingly.
- The current simulator has contact friction without tabletop drag. Do not expose a
  slider promising free-sliding slowing unless the simulator contract is changed and tested.
- Default demo controls stay inside the trained distribution. Mark deliberate OOD
  requests. Real-photo uploads are outside the core demo; synthetic uploads are deferred.
- Trace overlays come from the shared pixel tracker. Missing tracks remain visible as
  failures. Auxiliary-head trajectories, if shown, have a separate explicit label.

## Preset families

| Preset | What it tests |
|---|---|
| Free push | Force direction and magnitude on an isolated selected object |
| Direct collision | Motion propagates to an initially stationary second object |
| Glancing contact | Contact geometry and changes in direction/rotation |
| Near miss | Nearby objects do not react without contact |
| Hidden-world ensemble | Same frame/action with different reference masses or contact properties |
| Three-object chain | Only after the model supports this object count; otherwise an explicitly labelled OOD case |
| Failure example | A representative limitation, with its recorded run and explanation |

Choose presentation presets from a declared development/validation collection. Do not
select a final checkpoint or tune the system using final-test examples. Maintain a separate
untouched benchmark and report how presentation examples were chosen.

## A rehearsed 90-second demonstration

1. Explain the selected object and force; generate the four futures.
2. Show a direct-contact consequence on the other object.
3. Reverse the arrow using the same seed list; replay both results.
4. Switch models on the identical request and explain the measured comparison.
5. Reveal reference outcome spread and overlay generated endpoints.
6. Show one failure or the optional goal-region mode if it passed its gate.

The final timing depends on the instructor's slot and measured inference latency. Prepare
a shorter version. Do not promise a particular model wins before its results exist.

## Highest-ceiling extension: choose a push for a goal

Enable only after the required comparison and core demo are complete, the feature has
an owner, and its latency fits the presentation.
Keep it disabled until its acceptance checks pass; final test use must not tune it.

The user marks a target region for a named object. Generate a small fixed set of allowed
push candidates. A cheap CVAE rollout stage proposes a shortlist; diffusion evaluates
the shortlist. Rank using generated-pixel endpoints and sample validity, then show the
selected force and its predicted futures. The target can be the unselected object only
if the model has demonstrated contact propagation.

The simulator executes the chosen action afterwards for comparison. It never supplies
candidate outcomes to the search. Missing tracks count against a candidate, not as hidden
exclusions. Limit candidates to allowed inputs and record total sampling cost.

Evaluate a fixed held-out goal suite against random-action and CVAE-only search, using
the same realized simulator physics per trial. Report success, failures and latency.
"Three of four samples enter the region" is a sample count, not a calibrated 75%
success probability. Avoid collision-safety claims with the current contact detector.

Goal-conditioned physical video generation is already demonstrated by
[Goal Force](https://goal-force.github.io/). This optional search interface has a different
implementation; it is an application feature rather than a claim to invent goal planning.

## Inference, caching and deployment

- One loaded model service, bounded inference queue, deterministic per-request seeds
  and cancellation/progress. A late result cannot overwrite a newer scene/action request.
- Cache exact requests using model/checkpoint, visible input/mask, force, preprocessing,
  sampler settings, horizon and seeds. Do not key model predictions on hidden reference
  physics. Parameter changes invalidate the relevant cache.
- Distinguish live inference from cached model replay. No nearest-neighbour substitution
  or simulator-generated video is presented as a new neural prediction.
- Measure end-to-end warm median/p95 latency for four futures on the actual demo device,
  including conversion and display/export. Initial UX target is 15 seconds, not a result.
- Prefer a verified local checkpoint path on the presentation laptop. If inference uses
  TC2, run within an approved compute allocation and rehearse connectivity; never run GPU
  inference on its head node. Do not rely solely on classroom network/VPN availability.
- Prepare labelled local model replays and an embedded walkthrough as presentation
  fallback. Keep arbitrary unavailable controls disabled in replay-only mode.

## Demo acceptance

| Check | Pass evidence |
|---|---|
| Real model integration | The same checkpoint/sampler serves the UI and offline evaluation |
| Control | Visible held-out movement, reversed/zero/target controls and explicit stationary/failure counts |
| Object preservation | Trackable objects and coherent playback on the declared preset set; failures retained |
| Reproducibility | Repeated seeds on the same supported device reproduce the documented output tolerance |
| State correctness | Scene/force/model/seed changes, cancellation, repeated requests and errors behave correctly |
| Fair comparison | Common scene/action/horizon/seeds and clearly named methods/reference source |
| Latency | Recorded warm median/p95; live versus replay labels correct |
| Device independence | Full local rehearsal, checkpoint loading, embedded media and offline fallback verified |
| Presentation | A timed walkthrough, short version, one failure explanation and Q&A backup |

Use at least 20 scripted interaction sequences covering the controls above; functional
success does not replace model-quality measurements. Fix the acceptance preset set and
quality criteria before choosing final examples.

Record every selected example's checkpoint, condition, seeds, sampler and data provenance.
Keep model files and replay datasets out of the course source ZIP. Embed assessed figures
and results in the report and locally embed presentation media in the deck.

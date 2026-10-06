# Experiment and evaluation plan

## Hypotheses

- H1: spatial action conditioning improves action adherence over numeric-action diffusion when both identify the selected object.
- H2: trajectory/collision supervision improves physical behavior of the generated video over spatial conditioning alone.
- H3: repeated samples reflect outcome variation caused by hidden mass/friction, without relying on texture flicker or implausible motion.

Report negative results and uncertainty. The project does not assume any hypothesis will hold.

## Core comparison ladder

| ID | Model | Conditioning | Physics supervision |
|---|---|---|---|
| E0 | Conditional VAE | First frame, selected mask, numeric force | None |
| E1 | Latent video diffusion | First frame only | None |
| E2 | Latent video diffusion | First frame, selected mask, numeric force | None |
| E3 | E2 + object-centric spatial module | E2 inputs + spatial action map | None |
| E4 | E3 + auxiliary heads | Same as E3 | Trajectories + collision events |

E1 deliberately lacks action information; E2→E3 is the conditioning comparison. E3→E4 is the supervision comparison. If time permits, add E4-T (trajectory only) and E4-C (collision only); these are optional, not prerequisites for the main result.

Keep diffusion backbone, autoencoder, data splits, resolution, clip length, optimizer policy, training budget, sampler, and number of samples fixed where applicable. Record parameter counts, training steps, GPU-hours, and inference latency. Account for added module parameters and avoid claiming an architecture-isolated effect if model capacity also changes materially. Use independent training runs from a common initialization policy for the main E1-E4 comparisons. If one model is warm-started from another, report all inherited training and treat that result as a separate experiment.

## Test suites

The course minimum is two methods evaluated on at least one dataset. Our CVAE and diffusion comparison meets that structure; the planned E1-E4 variants provide component analysis. Use results from our runs, rather than copy values from reference papers.

| Suite | Change from training | Control |
|---|---|---|
| ID | New visible scenes and hidden physics draws | Training distributions |
| Shape OOD | Disjoint shape families | Known textures and force range |
| Texture OOD | Disjoint texture families | Known shapes and force range |
| Force OOD | Held-out magnitude range | Known shapes/textures; stable physics bounds |
| Uncertainty | Multiple hidden-physics rollouts per identical visible condition | Exact same first frame, mask, and action |
| Action controls | Zero force, reversed force, different selected object | Same visible scene |

Split by visible scene identity before generating variants. Freeze test sets before tuning. Decide exact ranges and sample counts in T04/T13 based on simulator stability and the measured compute budget.

## Metrics and interpretation

| Metric | Report | Caveat |
|---|---|---|
| Trajectory ADE | Mean Euclidean position error over future frames and valid objects | Use normalized coordinates; report selected-object and all-object scores |
| Trajectory FDE | Euclidean error at the final future frame | Separate from ADE |
| Action adherence | Selected-object initial direction error, stationary-output rate, and zero-force drift | Use a predefined interval before contact on controlled test scenes |
| Collision | Accuracy, precision, recall, F1; PR-AUC if probabilities exist | Include positive prevalence; accuracy alone can reward always predicting no collision |
| SSIM / LPIPS | Frame-aligned scores over future frames | Another valid physical future can score poorly against one reference |
| Temporal consistency | Motion-compensated image error plus trajectory acceleration/jitter checks | Raw frame differences penalize correct motion and reward static output |
| Diversity | Pairwise generated trajectory distances and collision-frequency variation | Pair with quality/coverage; visual noise is not useful uncertainty |
| FVD (optional) | Fixed encoder/preprocessing, equal counts, uncertainty estimates | Secondary only; do not rely on a tiny sample or silently changed feature extractor |

For sample position `p[k,t,o]` and reference `q[t,o]`, ADE is the average of `||p-q||₂` over future t and valid o; FDE uses the last future t. Report mean-over-samples and best-of-K separately with K fixed (initially 4). Best-of-K is an optimistic coverage statistic, not the expected prediction error.

Define `minADE@K` by taking the lowest whole-clip ADE across K complete samples. Do not pick a different sample for each frame or object. Define `minFDE@K` separately and also report the FDE of the sample selected by minimum ADE, so readers can distinguish the two selection rules.

For action-direction error, fix a nonzero-force control set with a visible-motion threshold and a short interval before any simulator contact across its physics variants. Measure the angle between the force and the selected object's displacement. Report stationary outputs and tracking failures separately; exclude them from the angle calculation but include them in the control-set denominator. For zero-force scenes, report displacement from the initially stationary positions. Fix the interval and thresholds on validation data.

Record the SSIM data range, window, and channel aggregation; record the LPIPS backbone, weights, input normalization, and any resizing. Score only predicted future frames, so copying the supplied first frame cannot improve the average. Use the [primary metric references](REFERENCES.md#metrics) when choosing implementations.

## Measure generated pixels, not only the auxiliary head

Build a shared object detector/tracker for synthetic rendered frames and validate it on held-out simulator videos before using it on generations. Establish object identity from frame 0, handle tracking failures explicitly, and report the failure rate. Do not remove hard samples without disclosing them. Apply the same extraction and tracking rules to every model.

Estimate contact events consistently from extracted masks/geometry, calibrate the rule against simulator labels, and report its limitations. Evaluate auxiliary collision/trajectory predictions separately for models that have those heads. A correct auxiliary head does not prove that the decoded video is physically correct.

Mask invalid object slots; score each unordered collision pair once. Fix a time tolerance (proposed: ±1 video frame) and a one-to-one event matching rule on validation data. Report object-wall contacts separately from object-object collisions.

Report interval-wise collision accuracy over valid unordered object-pair/time bins without temporal tolerance. Report event precision/recall/F1 using the stated one-to-one temporal matching rule. State the convention for no-positive cases; do not conflate event scores with interval accuracy. A video-derived contact estimate can miss a collision that occurs between frames, so quantify this error on simulator clips before interpreting model differences.

For temporal metrics, mask occluded or unreliable pixels in motion compensation and report the valid fraction. Score trajectory jitter away from contact intervals separately from collision motion, rather than penalize a sudden collision-induced velocity change as visual instability. Include a static-video control to expose metrics that reward lack of motion.

## Controlled uncertainty experiment

For each held-out condition, sample K=4 generated videos and multiple simulator futures (target at least 8 if affordable) from the known hidden-physics distribution. Report:

- Generated trajectory spread versus simulator trajectory spread.
- Coverage of reference outcomes by generated samples and distance of generated outcomes to the simulator reference set.
- Collision frequency and endpoint distributions for selected-object outcomes.
- Mean and best-of-K ADE/FDE to a fixed reference, clearly labelled as single-reference measures.
- Quality/action adherence alongside diversity, using the same sample seeds across variants.

Increase sample counts for a smaller diagnostic subset if four samples are insufficient to inspect distributions. A single realized simulator future is not the only physically valid answer when mass/friction are hidden.

## Fair reporting

- Choose hyperparameters and checkpoint criteria using validation data only.
- Target three training seeds for the main E2/E3/E4 comparisons if budget permits; state actual counts and limitations.
- Bootstrap uncertainty intervals over independent scene groups, not correlated rollout variants.
- Keep model sampling seeds separate from data and training seeds.
- Record dropped/corrupted samples, tracking failures, and all metric denominators.
- Show successful and failed examples, action controls, and OOD cases with run IDs.
- Never populate result tables with illustrative values that could be mistaken for measurements.

## Hyperparameter analysis for the report

Reserve a small validation study in T13/T14, since the course requires analysis of hyperparameter settings as well as components. A low-cost starting point is to compare a few diffusion sampling-step counts using the same trained checkpoint, conditions, K, and sampling seeds. Record quality, action adherence, and inference latency. If budget allows, add an auxiliary-loss weight study with separately trained models.

Choose tested values and the budget before running the study, select settings on validation data, and keep the final test protocol fixed. Include the study and its limitations in the report alongside E2-to-E3 and E3-to-E4 component comparisons.

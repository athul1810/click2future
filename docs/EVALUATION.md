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

Keep diffusion backbone, autoencoder, data splits, resolution, clip length, optimizer policy, training budget, sampler, and number of samples fixed where applicable. Record parameter counts, training steps, GPU-hours, and inference latency. Account for added module parameters and avoid claiming an architecture-isolated effect if model capacity also changes materially.

## Test suites

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
| Collision | Accuracy, precision, recall, F1; PR-AUC if probabilities exist | Include positive prevalence; accuracy alone can reward always predicting no collision |
| SSIM / LPIPS | Frame-aligned scores over future frames | Another valid physical future can score poorly against one reference |
| Temporal consistency | Motion-compensated image error plus trajectory acceleration/jitter checks | Raw frame differences penalize correct motion and reward static output |
| Diversity | Pairwise generated trajectory distances and collision-frequency variation | Pair with quality/coverage; visual noise is not useful uncertainty |
| FVD (optional) | Fixed encoder/preprocessing, equal counts, uncertainty estimates | Secondary only; do not rely on a tiny sample or silently changed feature extractor |

For sample position `p[k,t,o]` and reference `q[t,o]`, ADE is the average of `||p-q||₂` over future t and valid o; FDE uses the last future t. Report mean-over-samples and best-of-K separately with K fixed (initially 4). Best-of-K is an optimistic coverage statistic, not the expected prediction error.

## Measure generated pixels, not only the auxiliary head

Build a shared object detector/tracker for synthetic rendered frames and validate it on held-out simulator videos before using it on generations. Establish object identity from frame 0, handle tracking failures explicitly, and report the failure rate. Do not remove hard samples without disclosing them. Apply the same extraction and tracking rules to every model.

Estimate contact events consistently from extracted masks/geometry, calibrate the rule against simulator labels, and report its limitations. Evaluate auxiliary collision/trajectory predictions separately for models that have those heads. A correct auxiliary head does not prove that the decoded video is physically correct.

Mask invalid object slots; score each unordered collision pair once. Fix a time tolerance (proposed: ±1 video frame) and a one-to-one event matching rule on validation data. Report object-wall contacts separately from object-object collisions.

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

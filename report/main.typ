#set page(
  paper: "a4",
  margin: 22mm,
  header: context {
    if counter(page).get().first() > 1 {
      align(right, text(size: 8.5pt)[Click2Future | AI6132 | Group 7])
    }
  },
  footer: context align(center, text(size: 9pt)[#counter(page).display()]),
)
#set text(font: "Libertinus Serif", size: 11pt)
#set par(justify: true, spacing: 5pt, first-line-indent: 0pt)
#set heading(numbering: "1.")
#set figure(numbering: "1")
#show table: set text(size: 9pt)
#show figure.caption: set text(size: 9.5pt)

#align(center)[
  #text(size: 18pt, weight: "bold")[Click2Future: Action-Conditioned Video Generation for Tabletop Futures]
  #v(8pt)
  Group 7, AI6132 Generative AI For Visual Synthesis\
  #text(size: 9pt)[PRAMOD SANJAY, KAPOOR ANANYA,\ MAITI SHRIJEET, BOBAN ATHUL KRISHNA]
  #v(5pt)
  #text(size: 9pt)[13 November 2026]
]

#block(inset: (x: 10mm, y: 4mm))[
  #align(center)[*Abstract*]
  #set text(size: 9.5pt)
  We build Click2Future to generate and compare four futures from an image,
  a selected object and an applied force. A replayable tabletop simulator varies
  hidden mass and contact properties while holding the observed scene fixed.
  We compare a conditional video VAE with numeric- and spatial-conditioned pixel
  diffusion, using three training seeds and five held-out splits. The flow VAE
  preserves appearance and follows push direction over the full clip at the
  training force, although
  force magnitude and object selection remain unreliable. Longer pixel-diffusion
  training removes noise but does not preserve object identity reliably. Pose
  supervision and contact-aware training improve tracking, action response and
  rest behaviour. A geometric projection removes measured overlap, with
  exact-outline timing revealing a sub-pixel contact gap. On ten fresh
  in-distribution (ID) scenes,
  a dynamics-residual and momentum-correction hybrid reduces same-world
  trajectory error by 16.8%, mainly through the explicit correction; collision
  error increases and multi-reference energy scores still favour a prior-matched
  simulator. The interactive interface makes these comparisons inspectable
  through paired forces and synchronized futures. A separate photograph
  interface integrates external pretrained models. Together, the experiments
  distinguish improvements in appearance, action response and geometry from
  improvements in the predicted motion distribution.
]

= Introduction
<introduction>
An image reveals an object's visible shape and location, but not its mass,
friction or restitution. The same image and applied force can therefore admit
several valid futures. A useful generator must respond to the action, preserve
object identity and represent plausible variation in motion. Visual diversity
alone does not show that these conditions hold.

We predict future RGB frames from an initial image, a selected-object mask and
a force vector. Our controlled environment is a bounded, top-down tabletop
with two rigid objects. Exact simulator labels let us compare the same visible
scene under different hidden physics. This setting excludes depth, deformable objects and
arbitrary photographs. The photograph interface in @sec:demo uses externally
pretrained models and is evaluated separately.

The main comparison tests a conditional video VAE against two pixel diffusion
models with the same backbone: numeric force conditioning and a force map
localized to the selected object. The final VAE uses a flow decoder chosen on
validation; its RGB decoder remains an ablation. We then study pose supervision,
contact losses, sampled hidden physics and explicit geometric corrections.
@tab:models defines the model labels used throughout. D014-D039 identify
experiment records, not additional model families.

The project combines a replayable dataset, trained generators and an interactive
four-future comparison. Decoder studies test appearance preservation, while
paired controls test action routing, training losses and explicit motion corrections.
Multiple reference futures test how the generated samples represent variation
from hidden physics.

Variational autoencoders learn latent-variable models through an approximate
posterior @kingma; conditional VAEs extend this to structured outputs given an
input @sohn. Diffusion models learn to reverse gradual corruption @ho. Denoising diffusion
implicit models (DDIM) permit fewer sampling steps @song. We implement compact versions for
the tabletop task. Force Prompting @forceprompting and StreamForce @streamforce
provide image-conditioned force control at a much larger pretrained scale;
OneWorld @oneworld studies consistency of hidden physical mechanisms across
actions. Their tasks and training resources differ from ours, so we do not rank
these systems using our synthetic scores.

@methods describes the data and models. @experiments-and-analysis reports the
primary held-out comparison and the validation diagnostics behind it.
@sec:exploratory examines pose models, sampled physics and explicit geometric
corrections, and @sec:demo describes the interactive interfaces.

#figure(table(
  columns: (0.6fr, 2fr, 2.2fr),
  align: left,
  table.header([Label], [Model or change], [Training information]),
  [E0], [Conditional video VAE; final decoder warps frame-zero pixels], [Future RGB; extra displacement supervision is a separate ablation],
  [E1], [Action-blind pixel diffusion control], [Future RGB],
  [E2 / E3], [Numeric / spatial force conditioning in pixel diffusion], [Future RGB; same backbone and selected mask],
  [E4], [Auxiliary trajectory and contact heads on E3], [RGB, future positions and contacts],
  [E5 / E6], [Pose diffusion / contact-aware training], [Future rigid poses; E6 adds contact and rest objectives],
  [E7], [Relational pose model with a sampled physical world], [Pose, contact and physical-property labels],
  [E8 / E9], [Temporal residual / local surface messages], [Refinements of one E7 teacher],
  [E10], [Matched numeric-versus-spatial routing in pose space], [Pose and contact supervision; fresh initializations],
  [E11], [One-pass versus diffusion decoding after a sampled world], [Refinements of one E7 teacher],
), kind: table, caption: [Model labels. The main E0/E2/E3 comparison uses future RGB supervision. Later pose and physical-property supervision changes the task; those results are reported as separate studies.])
<tab:models>

= Methods
<methods>
== Simulator and data generation
<simulator-and-data-generation>
We generate clips with the CPU Pymunk rigid-body simulator and render RGB
frames and instance masks. The world is a unit square with zero gravity;
positive $x$ points right and positive $y$ points down. Objects start at rest
without overlap. The core corpus uses circles and rectangles; the
generalization corpus adds triangles and object-attached stripe or checker
patterns. There is no depth, stacking, lifting, deformation or surface drag.

A constant force acts at the selected object's centre for 0.25 seconds,
followed by unforced motion. It acts over time, rather than setting velocity,
and creates no off-centre torque. Mass, contact friction and restitution are
sampled independently of appearance from $[0.5,2.0)$, $[0.1,0.8)$ and $[0,0.4)$.
Pymunk multiplies the coefficients of contacting shapes. Object-object contact
can therefore involve friction and bounce; walls have zero friction and
restitution. There is no separate drag against the tabletop.

Clips contain 32 frames at 8 fps and $64 times 64$ pixels. Frame zero precedes
the force; the last future frame is at 3.875 seconds. Physics advances at 240 Hz
with ten solver iterations. A rollout stores RGB, masks, positions, valid object
slots, action metadata and contact onsets aggregated between frames. Scene
specifications, seeds and timing settings permit deterministic replay. In the
controlled corpus, the visible-scene seed also determines its action cases.

For each visible two-object scene, the controlled corpus covers six action cases: zero force, equal-and-opposite forward and reverse pushes, glancing collisions in both object-selection directions, and a push into a wall. The zero-force control has one canonical rollout; every nonzero action has four matched hidden-physics variants, giving 21 clips per ordinary scene group. Repeated-physics groups use eight nonzero profiles and contain 41 clips; force-OOD groups omit zero force and contain 20. Controlled-data validation checks that zero force remains stationary and contact-free, forward and reverse follow their requested directions without contact, both collision actions produce object contact without wall contact, and the wall action produces wall contact without object contact. Multiple references for one observable condition share frame zero, selection, force and timing while varying only hidden physics.

All actions and hidden-physics variants of one visible scene stay in one split.
The generalization corpus has disjoint training, validation, ID-test, shape-OOD,
texture-OOD, force-OOD and repeated-physics splits. ID means new scenes from the
training distribution; OOD changes the named factor. Generalization comparisons
use models trained on this corpus, rather than an earlier model tested on
unmatched data.

*Model observations and action encoding.* The observed inputs are frame-zero RGB, the selected mask and numeric force.
Future states and true hidden coefficients are unavailable at inference. RGB
is normalized to $[-1,1]$. Forces are divided by 0.5 without clipping; a separate
magnitude limit validates inputs. The spatial action map is
$A_c(u,v)=M(u,v)f_c, quad c in {x,y}$, where $M$ is the selected mask.
E2 and E3 receive the same mask, so their difference is force localization,
not whether the model knows which object was selected.

== Conditional video VAE
<conditional-video-vae>
E0 uses a conditional Gaussian prior $p_psi(z | c)$ and posterior
$q_phi(z | c,Y)$, where $c$ is the observed condition and $Y$ the future clip.
A 2D encoder processes the initial image, mask and broadcast force. A 3D
encoder sees future RGB during posterior training. The decoder receives the
condition features, one clip-level latent and normalized time coordinates.
The RGB variant predicts a tanh-bounded clip; the final flow variant warps
frame-zero pixels, as described in @cvae-decoder-follow-up.

The RGB pilot has 574,659 parameters, base width 16 and latent dimension 64.
Its objective is
$cal(L)_"VAE" = cal(L)_"RGB" + beta/d_z sum_(j=1)^(d_z) D_"KL"(q_(phi,j) || p_(psi,j))$,
averaged over the batch. Here $cal(L)_"RGB"$ is mean absolute reconstruction
error on future frames, with the foreground weighting reported for each run,
and $d_z$ is the latent dimension. Beta warms up with the saved training step.
Sampling draws from the conditional prior; the posterior is used only during
training and reconstruction diagnostics. Posterior reconstruction and prior
generation quality are therefore reported separately.

== Pixel video diffusion
<pixel-video-diffusion>
The diffusion model denoises RGB future frames directly. Its network combines
spatial and temporal convolutions, encoder-decoder skip connections and
temporal attention at reduced spatial resolution. The initial frame and action
channels are broadcast over time. Diffusion timesteps and numeric force also
provide conditioning embeddings.

For a clean future $Y$ and Gaussian noise $epsilon$, the forward process is
$Y_t=sqrt(macron(alpha)_t)Y+sqrt(1-macron(alpha)_t)epsilon$.
The code supports noise prediction or velocity prediction,
$v_t=sqrt(macron(alpha)_t)epsilon-sqrt(1-macron(alpha)_t)Y$,
with mean squared target error. The compared E2/E3 runs use the same velocity
target, schedule, width and initialization policy.

E1 replaces the selected mask and action with zero inputs. E2 supplies the selected mask and broadcasts numeric action fields over the image. E3 multiplies those fields by the selected mask, retaining the same backbone dimensions and numeric embedding. This is the implemented object-centric conditioning mechanism. E3 does not add a learned object detector or receive oracle future object states.

Inference uses an exponential moving average (EMA) of the weights and deterministic DDIM updates with private seed generators. Clean estimates are clipped to $[-1,1]$ and epsilon is recomputed after clipping. The exact original first frame is prepended to every generated future. A 16/32/64-step validation study is reported below; no setting passes the physical-video checks.

== Auxiliary supervision
<auxiliary-supervision>
E4 adds trajectory and contact heads to E3's denoiser features. Training labels
use a canonical frame-zero object order and valid-object/pair masks. Trajectory
heads predict heatmaps whose moments give positions; contact heads predict pair
logits. The configured timestep range and weights determine when these losses
apply. Head outputs are not inference inputs. We evaluate them separately from
motion and contacts extracted from generated RGB.



== Supervised object-trajectory diffusion
<supervised-object-trajectory-diffusion>
E5 separates stochastic motion from observed appearance. Frame-zero RGB and the
selected mask identify two visible objects, ordered by centroid independently
of the click. Each object receives its visible geometry and the force routed
through its selection indicator. A four-block, width-128 transformer with
845,828 parameters predicts 31 future poses per object, encoded as
$z_(t,i)=[Delta x/0.5, Delta y/0.5, sin(Delta theta), (cos(Delta theta)-1)/2]$.
Training uses EDM preconditioning and weighted denoising @karras plus a temporal
error-difference penalty. Sampling uses 16 Heun steps with rho 7, requiring
31 network calls per future. A rigid renderer translates and rotates observed
sprites; it does not solve contacts or correct sampled motion.

Future positions and replay-checked angles supervise training. Angular-label
replay is required to agree with saved trajectories within $10^(-6)$. Training
rotations and reflections transform images, masks, forces and poses together.
Hidden physical coefficients remain unavailable at inference. Pose supervision
and rigid rendering distinguish E5 from the RGB-only E2/E3 comparison. The
method requires two separately visible objects in the synthetic image domain.

== Contact-aware trajectory training
<contact-aware-trajectory-training>
E6 retains E5's network, observable inputs, rigid renderer and sampler, while changing the training package. Convex hulls extracted from frame-zero masks provide differentiable separating-axis geometry. An excess-penetration loss penalizes predicted depth beyond one native pixel; a contact-gap loss compares predicted and labelled separation near reference contact. A stationary-motion loss supervises rest, and an acceleration-error term supplements temporal error. Derivative losses compare with labelled motion, allowing changes at impact. Future poses identify supervision during training only. Sampling has no contact projection, rest override or motion correction.

The first attempt balances zero-force, contact and other training examples. The second also supervises unchanged object tokens, raises geometry weights and extends the noise range for geometric losses. The third retains that recipe and adds paired training actions plus more updates. These are combined changes, not separate causal ablations. All three start from scratch with seed 19; fixed-seed validation denoising loss selects checkpoints. Physics scores do not select among checkpoints, although the successive recipes respond to earlier validation failures.

= Experiments and analysis
<experiments-and-analysis>
== Evaluation protocol
<frozen-evaluation-and-failure-handling>
Evaluation plans fix the conditions, references, sample seeds and scorer
settings before sampling. Input and checkpoint hashes detect changed files;
the runner rejects incompatible training corpora and incomplete exports.
Each condition receives four sampled futures. Missing tracks stay in the
reported counts, even when a trajectory average cannot be computed.

A palette-based tracker identifies objects from frame zero. Blur, changed
colour or shape, occlusion and unfamiliar textures can break its tracks. A
complete all-object clip has every valid object tracked in every future frame.
Conditional averages use complete trajectories only and are accompanied by
their available-sample counts.

Average displacement error (ADE) averages position error over future frames;
final displacement error (FDE) uses the final frame. We report all-object and
selected-object errors, keeping the mean over samples separate from the
best complete clip among four. Structural similarity (SSIM) and learned
perceptual image patch similarity (LPIPS) @ssim @lpips compare appearance with
one declared reference. Contact scores distinguish any encounter, exact
contact intervals and one-to-one onset matches within one interval. Ensemble
spread and nearest-sample coverage use multiple hidden-physics references.
Trajectory distances are in normalized image coordinates unless stated
otherwise; percentage changes use unrounded scores. Later studies also report
same-world ADE against a simulator using the model's sampled physical
coefficients. That conditions on a latent draw and differs from first-reference
error or overall distributional accuracy.

Aggregation averages dependent samples and conditions within each visible
scene and training run, then averages runs and scenes. Confidence intervals
resample whole scenes and condition on the trained checkpoints; they do not
represent uncertainty from new training runs. One scene gives no bootstrap interval. The final
E0/E2/E3 comparison uses three training seeds per method (@sec:final).

The original engineering acceptance rule requires at least 90% tracked
all-object observations, complete selected-object tracks in every sample and
mean selected ADE at most 0.03125 (two native pixels) in every condition.
Action response and reference coverage are separate checks. This
single-reference rule can reject a valid future drawn from different hidden
physics; later simulator controls and energy scores examine that limitation.

== Action tests and contact measurement
<action-controls-and-extractor-calibration>
Action controls first score the initial 0.5 seconds, ending at frame four.
Nonzero-force motion is the selected object's endpoint displacement, with at
least one native pixel counted as moving. Zero-force drift is the maximum
excursion of any valid object. Every reference must start at rest and avoid
contact during the scored window; eligibility is fixed from references.
A supplementary score extends the same rules to 3.875 seconds where all
references remain contact-free. Force-OOD has no zero-force case but still
receives forward, reverse and reversal scores.

Selected-object controls require a separate intervention: the initial image, physical objects, force vector and duration remain identical under two different clicks. Derived references live outside the training corpus. Correct target following requires selected motion along the force and no above-threshold excursion by unselected objects in both conditions. A separate per-object contrast rejects identical motion under both clicks. The E1 control produces identical videos for both selections when seeds match, as expected for an action-blind model.

On two validation scenes, force 0.15 leaves all eight exact-label target pairs
below the one-pixel criterion, while rasterization moves some centroids past it.
A separate force-0.4 selection probe passes both label and rendered-reference
controls without changing the threshold. Static and ignored-click controls
fail. We retain the native-force calibration and use the validation-chosen
0.4 probe for selected-object tests.

Contact extraction has an independent timing limitation. On a fresh 63-clip controlled check over three scene groups, all 3,906 object observations were tracked. Any-encounter scoring produced TP=23, FP=0, FN=1 and TN=39, while timed-event F1 was 0.551. The stronger occurrence result cannot substitute for the weaker event-timing result. These extractor measurements do not count the 63 dependent clips as 63 independent scenes.

== Held-out RGB generation
<sec:final>
The primary held-out comparison fixes E0 as the flow CVAE and uses training
seeds 0-2 for every method. Each E0 run completes 20,000 batch-two updates;
validation selects checkpoints at updates 20,000, 14,000 and 13,000 for seeds
0, 1 and 2. E2/E3 retain 5,000 batch-one updates, final EMA
weights and 32 DDIM steps; the validation step study found no successful
alternative. Their six validation runs track only 1-546 of 5,952 observations
and produce no complete clip. We nevertheless evaluate all five reserved
splits to report held-out failures. Checkpoints and settings are fixed before
those results are inspected.

The primary evaluation prepares ten plans before sampling: one ordinary and
one force-0.4 selection plan for each split. Nine trained runs produce four
futures for each of 58 ordinary conditions, with 248 simulator references.
Force-OOD contains forces 1.3 and 1.5 and no zero-force condition; reversal
is still scored. Every suite completes.

#figure(
  align(center)[#table(
    columns: 7,
    align: (left,right,right,right,right,right,right,),
    [], table.cell(align: center, colspan: 3)[Complete samples], table.cell(align: center, colspan: 3)[SSIM / LPIPS],
    [Split], [E0], [E2], [E3], [E0], [E2], [E3],
    [ID], [136/144], [0/144], [0/144], [0.980 / 0.014], [0.465 / 0.220], [0.458 / 0.229],
    [Shape OOD], [90/144], [0/144], [0/144], [0.966 / 0.029], [0.475 / 0.248], [0.466 / 0.244],
    [Texture OOD], [123/144], [0/144], [0/144], [0.974 / 0.017], [0.462 / 0.178], [0.457 / 0.177],
    [Force OOD], [89/120], [0/120], [0/120], [0.886 / 0.144], [0.217 / 0.391], [0.256 / 0.332],
    [Repeated physics], [132/144], [0/144], [0/144], [0.973 / 0.012], [0.457 / 0.180], [0.451 / 0.183],
  )]
  , caption: [Primary held-out evaluation. Three training seeds per method and four sampling seeds per condition. Complete samples have every object tracked in every future frame. SSIM/LPIPS are single-reference scores pooled over training seeds.]
  , kind: table
  )
<tab:final>

E2 and E3 have no complete sample on any split, so their trajectory, collision and action scores are undefined rather than zero. They keep the scene layout but leave heavy noise in every generated frame (@fig:final), and the tracker loses most objects. Outside force OOD their SSIM is 0.45-0.48 against E0's 0.97-0.98, and their LPIPS is 8 to 16 times E0's. The E2-versus-E3 comparison therefore cannot be made on physical behaviour. Their SSIM differs by at most 0.04, with E2 slightly higher on four splits and E3 on force OOD. With two scene groups per split, the bootstrap intervals are not informative. Selected-object ADE and FDE are undefined for every method, because no scene group has all its cases complete.

E0's action response on the held-out splits repeats its validation behaviour (@sec:horizon). On the ID, shape, texture and repeated-physics splits the correct reversal-sign rate is 0.00-0.08 in the 0.5-second window, below the exact-label and rendered references (0.25-0.50), but 1.00 over the full clip. Every training seed moves all eight of its forward and eight of its reversed samples on each of these splits. Zero-force drift over the full clip occurs only in two samples of seed 0 in one repeated-physics scene, at 2.2 and 1.4 pixels, while the median drift on each split is at most 0.009, below one pixel. E0 therefore moves the selected object in the requested direction, but too slowly for the window.

Force OOD breaks this pattern. Within the window its references move clearly, while E0 moves the 24 reversed samples by a median of 0.081 (about five pixels) and the 24 forward samples by only 0.005, so no forward sample reaches the one-pixel threshold. This asymmetry is consistent with the training split: its only strong force is the wall action, 1.1 towards the left, the same direction as a reversed push, while forward pushes are trained only at 0.15 towards the right. A strong leftward push resembles a trained case, whereas a strong rightward push does not; this association is not a causal ablation. The full clip is not scored on force OOD because every group reaches contact first.

Selected-object control fails on every split. No complete pair moves the correct target in the window. Over the full clip the only scorable pairs, eight in one ID group, show the validation pattern in @sec:horizon more strongly. When object 0 is selected it moves by a median of 0.115 while the other object stays below one pixel in all eight pairs. When object 1 is selected it moves only 0.009, while object 0 drifts by a median of 0.053, above one pixel in all eight. Training selects object 1 only when it is pushed into object 0, so E0 moves object 0 even when object 1 is selected.

Contact occurrence is predicted better than contact timing. Collision event F1 is 0.09-0.22, while encounter F1 is 0.92-0.95 on the ID, texture and repeated-physics splits, 0.75 on force OOD and 0.31 on shape OOD. E0 produces no false encounter on four splits and one on force OOD, so its encounter errors are missed contacts, and the low shape-OOD score reflects 2 of 11 contacts found. Its event F1 should be read against the calibration in @tab:collision_calibration, where valid alternative physics reaches only 0.24 against a single reference.

== Longer diffusion training
<sec:d015>
In @tab:final, E2/E3 train on 5,000 clips per run, while each E0 run processes
40,000 before checkpoint selection. After the test results, we retrained E2/E3
for 40,000 updates with seeds 0 to 2, extending the cosine schedule across the
new limit. This matches the full per-run clip budget, not the exposure of each
selected E0 checkpoint, optimizer updates or GPU time. The seed-0 diffusion
runs take about 33 minutes on an A40; seeds 1 and 2 take about 19 minutes on
an L40S. On validation, seed 0's tracked observations rise from 176 (E2) and 5
(E3) at 5,000 updates to 3,434 and 2,869 at 40,000. Pooled over three seeds,
validation SSIM rises from 0.45-0.46 to 0.93. The longer runs complete 0 to 8
of 96 clips per run, against E0's 80 to 92.

This longer-training experiment was chosen after the primary test results,
so it is a secondary result. The 40,000-update checkpoint was fixed before
training, with no score-based selection. E0 is resampled on the same A40 type
and reproduces all 522 earlier report files. Validation tracking for the same
diffusion checkpoints changes by up to 208/5,952 observations on an L40S,
showing that exact numerical reproduction also depends on GPU type.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,right,right,right,right,),
    [], table.cell(align: center, colspan: 2)[Complete samples], table.cell(align: center, colspan: 2)[SSIM / LPIPS],
    [Split], [E2], [E3], [E2], [E3],
    [ID], [0/144], [10/144], [0.933 / 0.057], [0.933 / 0.059],
    [Shape OOD], [0/144], [0/144], [0.938 / 0.093], [0.939 / 0.088],
    [Texture OOD], [11/144], [15/144], [0.937 / 0.049], [0.937 / 0.048],
    [Force OOD], [0/120], [0/120], [0.744 / 0.188], [0.788 / 0.155],
    [Repeated physics], [7/144], [10/144], [0.934 / 0.034], [0.932 / 0.036],
  )]
  , caption: [Secondary held-out evaluation of E2/E3 trained for 40,000 updates. Same splits, plans, scorer and sampling seeds as @tab:final\; E0 is unchanged from that table.]
  , kind: table
  )
<tab:d015>

At 40,000 updates, the diffusion models lose the noise. Outside force OOD, SSIM rises to 0.93-0.94, within 0.03-0.05 of E0, and LPIPS falls to 0.03-0.09. Physical behaviour still fails. E2/E3 complete 0 to 15 of 144 samples per split and track 21 to 67 percent of object observations per run, against 88 to 100 percent for E0. Most remaining tracking failures are area changes and colour mismatches. In @fig:final, both models turn the small blue ball into a teal square by two seconds.

Pooled over training seeds, no complete reversal pair is correct in the 0.5-second window: 0 of 59 for E2 and 0 of 65 for E3. Over the full clip the few complete pairs mostly reverse correctly, 7 of 10 and 10 of 11. On repeated physics, however, 13 of 14 complete zero-force samples drift, against 2 of 24 for E0. Selected-object control fails. The only correct targets are E3's 2 of 11 complete window pairs on force OOD, and E2/E3 have no complete full-clip pair. E2 and E3 have the same SSIM within 0.01 except on force OOD. Their LPIPS intervals separate only on force OOD, where two scene groups make the interval uninformative. Over the samples whose selected track is complete, E2/E3 selected-object ADE is 0.008-0.035 in normalized image units. These samples are not representative: none of the E2/E3 complete pair timelines contains a contact in the reference or the generated video, so collision F1 stays undefined.

Longer training largely repairs diffusion's image noise, but not object
identity. Even at the full 40,000-clip training budget, neither E2 nor
E3 produces enough complete trajectories for a reliable physical comparison.
The numeric-versus-spatial conditioning question therefore remains unresolved
for these pixel models.

== Tracking failures and computational cost
<sec:cost>
@tab:split_tracking gives the share of future object observations the tracker loses on each split, pooled over training seeds. For E0 and the 40,000-update models, new shapes and new forces raise tracker failures, while new textures and repeated hidden physics stay close to ID. The 5,000-update models lose 86 to 100 percent everywhere. @fig:ood shows the first forward condition of each OOD split. On shape OOD, both 40,000-update models turn the large triangle into a square, and E0 rounds its corners. On force OOD, the reference pushes the rectangle to the wall within two seconds. No model moves it there: E0 leaves dark streaks at the image border, and E2/E3 smear the rectangle.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left,right,right,right,right,right,),
    table.header([Split], [E0], [E2, 5k], [E3, 5k], [E2, 40k], [E3, 40k],),
    table.hline(),
    [ID], [2%], [86%], [87%], [47%], [48%],
    [Shape OOD], [5%], [100%], [100%], [71%], [61%],
    [Texture OOD], [3%], [92%], [93%], [48%], [48%],
    [Force OOD], [6%], [97%], [99%], [73%], [69%],
    [Repeated physics], [2%], [100%], [99%], [46%], [46%],
  )]
  , caption: [Untracked future object observations on the reserved splits, pooled over three training seeds. The 5k diffusion models are those of @tab:final and the 40k models those of @tab:d015.]
  , kind: table
  )
<tab:split_tracking>

@tab:cost reports training and sampling times. At the 40,000-clip training
budget, diffusion takes about 2.6 times as long as E0 on an A40. Each video
needs 32 denoising steps, so in the same job they sample 42 (5k) and 47 (40k)
times more slowly than E0. The final-phase TC2 runs used 4.9 GPU-hours.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,right,right,right,right,),
    table.header([Method], [Parameters], [Updates], [Training], [Sampling per video],),
    table.hline(),
    [E0 flow CVAE], [574,226], [20,000 at batch two], [742-757 s], [0.012-0.015 s],
    [E2/E3, 5k], [964,035], [5,000 at batch one], [246-281 s], [0.54 s],
    [E2/E3, 40k], [964,035], [40,000 at batch one], [1,965-1,970 s], [0.58-0.59 s],
  )]
  , caption: [Wall-clock times from TC2 job records. Diffusion training times shown are the seed-0 A40 runs; the 40k seed-1/2 runs use L40S GPUs. E0 seed 0's GPU type is unrecorded. Updates denote completed training, not selected checkpoint steps. Sampling uses A40 test jobs and includes checkpoint loading and 32-frame video export.]
  , kind: table
  )
<tab:cost>

#figure(placement: none,
  image("figures/final_heldout.png", width: 86%),
  caption: [
    First sorted ID-test forward condition, sample seed 0 and training seed 0, at six fixed times, with one simulator reference. The 5,000-update diffusion models leave heavy noise in every generated frame. At 40,000 updates the noise is gone, but the small ball becomes a square and changes colour, so the tracker loses it. Selection is fixed before visual inspection; this is not a best-sample comparison.
  ]
)
<fig:final>


#figure(placement: none,
  image("figures/final_ood.png", width: 100%),
  caption: [
    First sorted forward condition of the shape-OOD, texture-OOD and force-OOD splits, sample seed 0 and training seed 0, at three fixed times, with one simulator reference. The diffusion models are the 40,000-update runs. Selection is fixed before visual inspection; this is not a best-sample comparison.
  ]
)
<fig:ood>




== Validation diagnostics
<multi-scene-validation-pilot>
The following validation studies explain the decoder choice and the failures
seen in the main comparison. They do not change its held-out results.

Dataset-0.3 seed 730000 contains 16 training scenes (336 clips), four validation scenes (84 reference clips) and 248 reserved test clips across the ID/OOD/physics splits. All 668 clips replayed exactly before training. Validation has six observable action conditions per scene, including zero, forward, reverse, both collision directions and a wall interaction. Four sample seeds per condition give 96 videos per method. Repeated hidden-physics references support coverage analysis; image distances use one explicitly declared reference per condition.

The RGB CVAE trains for 20,000 updates at batch two and seed zero; E2/E3/E4
train for 5,000 updates at batch one and seed seven. Diffusion uses width 32,
FP32 velocity targets, EMA 0.995, 1,000 noise steps and 32 DDIM steps. The CVAE
uses width 16, latent size 64 and beta 0.001 with a 20% warmup. Both losses
weight foreground pixels by 32 with a two-pixel margin, using future masks
only in the training loss. Validation posterior loss selects CVAE update
16,000; diffusion uses final EMA weights. Training and evaluation take 2,087
and 372 allocated seconds on an A40.

Diffusion runs start without inherited weights. E2/E3 each have 964,035
parameters; E4 adds 2,580 head parameters. Initializing those heads consumes
extra random draws, so equal E3/E4 seeds do not give identical training noise.
Training labels contain 156 positive and 10,260 negative pair intervals,
setting the capped positive weight to 20. One training seed, four validation
scenes and unequal CVAE/diffusion budgets limit this pilot; @sec:final reports
the primary held-out comparison and @sec:d015 the longer-training follow-up.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,right,right,right,right,),
    table.header([Method], [SSIM], [LPIPS], [Tracked / 5,952], [Complete / 96],),
    table.hline(),
    [E0 RGB CVAE], [0.9291], [0.1230], [398], [0],
    [E2 numeric diffusion], [0.4736], [0.1503], [378], [0],
    [E3 spatial diffusion], [0.4806], [0.1493], [269], [0],
    [E4 auxiliary diffusion], [0.4711], [0.1537], [162], [0],
    [Static frame-zero copy], [0.9446], [0.0622], [n/a], [n/a],
  )]
  , caption: [Validation pilot on 24 conditions and four scenes. SSIM is higher-is-better; LPIPS is lower-is-better. Image scores exclude frame zero and average cases within scene, then scenes. Tracking uses tracker version 2; complete means all valid future object tracks. Static copying is an appearance control, not a successful future predictor.]
  , kind: table
  )
<tab:multiscene>

No method has a complete selected-object sample, so mean and best-of-four ADE/FDE, trajectory spread and reference coverage are undefined. Undefined values remain missing rather than becoming zero error. All 16 forward, reverse and zero-action samples per method lack the required selected tracks; all 16 selected-object intervention pairs also fail tracking. These outcomes do not support an action-adherence or meaningful-uncertainty claim.

The whole-scene 95-percent LPIPS bootstrap intervals are \[0.0797, 0.1746\] for E0, \[0.1404, 0.1601\] for E2, \[0.1394, 0.1579\] for E3 and \[0.1425, 0.1649\] for E4. They are conditional on the supplied checkpoints and do not include training-seed uncertainty. The static-copy LPIPS interval is \[0.0460, 0.0783\]. Its stronger appearance scores expose background and displacement sensitivity: good single-reference image distance does not establish useful physical prediction.

The RGB CVAE's validation posterior MAE is 0.02091 and prior MAE 0.02113 in normalized RGB units. Training posterior/prior errors are 0.01205/0.01386. Its validation mean per-latent KL is 0.2741 and pairwise prior RGB MAE is 0.00330. Nonzero KL and sample variation do not establish useful latent uncertainty; these diagnostics alone also do not justify calling the failure posterior collapse.

=== Auxiliary heads versus generated RGB
<auxiliary-heads-versus-generated-rgb>
The E4 analysis replays all 96 validation samplings while capturing the actual final timestep-zero DDIM heads. Every RGB pixel matches the saved sample. Head predictions have all-object ADE 0.1314 and FDE 0.1560 in normalized coordinates against one declared reference. At the fixed 0.5 probability threshold, timed-contact matching with one-interval tolerance gives TP=28, FP=331 and FN=36, hence event F1=0.1324. Exact-interval positive prevalence is 64/2,976 and interval F1=0.1087. Any-encounter F1=0.5714 does not establish correct timing.

Only 162 of 5,952 future object observations are jointly available from the head and RGB extraction. Their mean position disagreement is 0.1481. No pair interval is jointly observed, so RGB/head collision agreement is undefined. Accurate-looking auxiliary outputs cannot substitute for decoded-video behaviour.

A separate 5,000-update check on the first sorted training collision clip has head ADE 0.00493 and contact-event F1 1.0 (TP=4, FP=0, FN=0 across four samples). RGB tracks 247/248 observations, with three complete clips. One missing selected observation still fails the strict gate; conditional selected ADE 0.00396 over the other three clips is not a best-of-four score. This training-only result shows that the head can learn a known contact, not that physical behaviour generalizes.

=== Sampling steps
<validation-sampling-step-study>
A predeclared study holds the E3 checkpoint, sample seeds and metric settings fixed. It uses the first four sorted validation scenes and zero, forward, reverse and collision-0-to-1 conditions, for 16 conditions and 64 videos per setting. Excluded validation conditions remain listed. This is a subset study rather than the full 24-condition comparison.

#figure(
  align(center)[#table(
    columns: 4,
    align: (right,right,right,right,),
    table.header([DDIM steps], [Export seconds], [Tracked / 3,968], [SSIM],),
    table.hline(),
    [16], [16.25], [330], [0.5473],
    [32], [29.37], [220], [0.5083],
    [64], [57.59], [184], [0.4931],
  )]
  , caption: [Same frozen E3 checkpoint on the fixed validation subset. Export time includes checkpoint loading and writing all 64 videos on a TC2 A40. It is not a pure denoiser latency measurement.]
  , kind: table
  )
<tab:steps>

All three settings have zero complete selected-object clips and fail the action checks through missing tracks. More sampling steps increase cost without repairing this checkpoint. The study does not select a successful sampler or establish a general advantage for fewer steps.

== CVAE appearance and action response
<cvae-decoder-follow-up>
The RGB samples change object appearance on held-out scenes. A separate exploratory CVAE therefore predicts a backward displacement field and bilinearly samples observed frame-zero pixels. Its zero-initialized output copies the initial image. The latent distributions, data, seed, RGB/KL loss and 20,000-update limit are retained, but decoder parameterization and initialization change. The model has 574,226 parameters. It cannot reconstruct unseen textures, enforce rigidity or enforce collision laws.

This flow model tracks 5,654/5,952 observations (94.99 percent), with 80/96 complete clips. Its single-reference SSIM is 0.9693 and LPIPS 0.0227. Nevertheless all 16 forward samples, all 16 reverse samples and all 16 selected-object intervention pairs are stationary in the fixed action window. Correct force-sign and target-following rates are zero over complete eligible pairs. Native-force interpretation is limited: 11/16 continuous and 8/16 rendered reference pairs also fall below the short-window motion threshold. In contrast, the separate force-0.4 selection probe passes all 16 continuous and rendered reference pairs, while this CVAE passes none. Better appearance is therefore insufficient.

A further displacement-supervised variant is kept separate from the ordinary CVAE: future masks and positions enter a training loss only, not generator conditioning. The displacement-supervised run completes 20,000 updates, with its combined validation criterion selecting update 9,000. It tracks 5,765/5,952 observations and 80/96 complete clips, with SSIM 0.9729 and LPIPS 0.0168. Yet only one of 16 reversal pairs has the correct force signs: 15 forward samples move, while only one reversed sample moves. All 16 selected-object pairs remain stationary. Additional supervision improves these appearance scores but does not establish reliable action control. Neither exploratory variant provides an independent final test.

=== KL weight, latent size and force response
<sec:cvae_study>
The RGB CVAE's action controls are undefined rather than failed: every forward, reversal and selected-object sample loses at least one track. The flow variants keep tracks but remain stationary in the 0.5-second window. Two explanations could leave a CVAE without usable action response. The conditioning path could discard the force, but it does not at initialization: across 20 randomly initialized networks on synthetic two-object scenes, flipping the force changes the prior's input more than switching the selected object, and about 90 percent of the force-induced change survives the first group normalization.

Alternatively, training could route motion through the posterior latent. The decoder can read motion from $z ~ q_phi(z | c,Y)$, while the prior is only weakly pulled towards the posterior under $beta = 10^(- 3)$.

To test this, we retrain the RGB CVAE on the same frozen corpus with $beta in {10^(-4),10^(-3),10^(-2)}$ and latent dimension in ${32,64}$, changing nothing else. Each run keeps the 20,000-update and 0.5-hour limits and the 20-percent beta warmup, selects its checkpoint by full validation posterior loss plus KL, and is scored with the same validation protocol and force-0.4 selection probe as the decoder follow-ups. The $beta = 10^(- 3)$, dimension-64 run repeats the original pilot on a separate TC2 account.

A force probe locates where each trained model carries the action. For every nonzero-force validation clip, we negate the force and decode while holding either the condition features or the prior mean fixed. The resulting changes inside the true future object region are compared with the spread between two prior samples and with the change from replacing the prior mean by the posterior mean. Future masks only choose the scored region; they never enter the model. A force effect that is small relative to the posterior-prior change indicates motion encoded in the posterior latent that the prior cannot recover from the action.

All six RGB settings run for 20,000 updates, about 12 minutes each on an A40,
using 0.22 GB peak tensor memory. Validation selects updates 15,000-18,000.
The beta-0.001, dimension-64 repeat reproduces update 16,000 and the original
398/5,952 tracked observations. No setting completes an all-object clip.
Dimension 32 tracks 27, 85 and 56 observations at beta 0.01, 0.001 and 0.0001,
versus 314, 398 and 365 for dimension 64. Increasing beta lowers mean KL to
0.07-0.08 and leaves 12/32 or 30/64 dimensions above 0.01; decreasing beta
raises KL to about 1.0 with every dimension active. Posterior and prior MAE
remain within 0.0202-0.0216 and 0.0211-0.0235. Only beta 0.0001 with dimension
64 yields a complete reversal pair: one stationary and one moving with the
wrong sign. Neither KL weight nor latent size repairs sampled-video quality.

The force probe does not support the latent-routing explanation. Negating
force changes the decoded object region by 0.169-0.178 mean absolute RGB error
across the six runs, compared with 0.124 between the true future and a static
frame-zero copy. The condition-feature path accounts for 0.168-0.177; changing
only the prior mean gives 0.008-0.024. Replacing the prior mean with the
posterior mean gives 0.048-0.056. Thus force affects RGB strongly without
producing trackable motion. Changing beta or latent size did not repair that
failure in these single-seed validation runs.

A repeated flow run reproduces 5,654/5,952 tracked observations, 80/96 complete
clips and posterior/prior MAE 0.00429/0.00604. Negating force changes the object
region by 0.150 mean absolute RGB error. The condition-feature intervention
gives 0.152, compared with 0.020 through the prior mean. Thus the flow decoder
also responds to force in pixel space, despite its short-window stationarity.


=== Motion over the full clip
<sec:horizon>
We score the saved validation videos at 0.5 and 3.875 seconds using the same
action rules. Forward and reverse reference pairs in all four validation groups
remain contact-free throughout, so both horizons use the same cases.
@tab:horizon includes the original flow
run, its two training-seed repeats and the displacement-supervised variant.

Frame-by-frame tracking explains the flow decoder's short-window result. At
force 0.15, reference displacement at 0.5 seconds averages 0.013 normalized
units; 11 of 16 forward references fall below the one-pixel threshold 0.0156.
The flow decoder averages 0.010. By 3.875 seconds, however, its forward and
reverse displacements reach 0.107 and 0.098 along the force, versus 0.133 for
references. Zero-force displacement is at most 0.001 on average.

The displacement-supervised variant averages 0.018 forward and 0.009 reverse
at 0.5 seconds, then 0.116 and 0.112 at the final frame. The diagnostic's
strictly-greater-than-one-pixel count differs from the official scorer, which
includes exactly one pixel; @tab:horizon uses the official counts.

At force 0.4, selected-object displacement is 0.071 for the ordinary flow
decoder and 0.118 for the supervised variant, versus 0.307 for references.
The unselected object moves 0.020 and 0.055. Training uses one magnitude per
action type: 0.15 for free pushes, 0.25 for collisions and 1.1 for walls, so
0.4 tests an unseen free-push magnitude. Selection is also confounded in
training: free pushes always select object 0, while selecting object 1 always
involves collision with object 0. Consistent with this association, object 0
drifts in 13/14 and 16/16 complete samples when object 1 is selected, but the
other object stays below one pixel in all 32 samples when object 0 is selected.
These observations identify coverage gaps; they do not isolate data bias from
decoder limitations.

Two additional flow training seeds reproduce the behavioural pattern. Across
seeds 0-2, 5,654, 5,916 and 5,947 of 5,952 observations are tracked, with 80,
92 and 91 complete clips. Full-clip forward displacements are 0.107, 0.109 and
0.107; reverse displacements are 0.098, 0.122 and 0.112, versus 0.133 for
references. Mean zero-force displacement stays below 0.003. At selection-probe
force 0.4, the selected object moves only 0.071, 0.090 and 0.112 against 0.307
for references. Selecting object 1 makes object 0 drift by more than one pixel
in 13/14, 14/14 and 6/6 complete samples; seed 2 loses a later track in ten of
those 16 samples. These repeats support the slow-motion and selection-bias
diagnosis beyond the original training seed.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,right,right,right,right,),
    [], table.cell(align: center, colspan: 2)[Forward / reverse moving], table.cell(align: center, colspan: 2)[Correct reversal signs],
    [Source], [0.5 s], [Full clip], [0.5 s], [Full clip],
    [Reference labels], [5/16 / 5/16], [16/16 / 16/16], [0.31], [1.00],
    [Reference renders], [8/16 / 8/16], [16/16 / 16/16], [0.50], [1.00],
    [Flow CVAE, seed 0], [0/16 / 0/16], [16/16 / 16/16], [0.00], [1.00],
    [Flow CVAE, seed 1], [0/16 / 10/16], [16/16 / 16/16], [0.00], [1.00],
    [Flow CVAE, seed 2], [0/16 / 3/16], [16/16 / 16/16], [0.00], [1.00],
    [Flow CVAE, displacement loss], [15/16 / 1/16], [16/16 / 16/16], [0.06], [1.00],
  )]
  , caption: [Validation action controls at native force, four scene groups. Rates are over complete pairs. Zero-force drift is 0/16 for every flow run at both horizons. E0 RGB, E2, E3 and E4 have no complete forward or reverse samples at either horizon, so their scores stay undefined.]
  , kind: table
  )
<tab:horizon>

The full-clip score therefore separates slow but correct motion from absent motion, while untrackable videos stay unscored. At the final frame only two of the four selected-object groups are contact-free at force 0.4. On those eight pairs, flow CVAE seeds 0, 1 and 2 complete six, eight and two, and the displacement-supervised variant eight, but every complete pair fails because the unselected object moves at least one pixel. Selected-object control therefore remains a failure for both flow models. Following these validation results, E0 is the unsupervised flow CVAE with training seeds 0 to 2. The RGB decoder is kept as a decoder ablation, and the displacement-supervised variant stays separate because it uses future positions during training.



#figure(placement: auto,
  image("figures/multiscene_validation.png", width: 100%),
  caption: [
    First sorted validation forward condition, seed zero, at six fixed times. The simulator is one declared reference. The E0 row is the RGB CVAE pilot; the separately shown Flow CVAE is the final E0 decoder. The RGB CVAE uses its conditional prior; E2/E3/E4 use EMA and 32 DDIM steps. Both flow variants preserve appearance but fail the selection controls in the 0.5-second window and over the full clip. The supervised variant also fails 15/16 force-reversal pairs in the window but passes all 16 over the full clip (@sec:horizon). Selection is fixed before visual inspection; this is not a best-sample comparison.
  ]
)
<fig:multiscene>




== Predicting poses instead of pixels
<trajectory-model-and-contact-audit>
#block(breakable: false)[
E5 trains for 6,000 updates on the same 336 training and 84 validation clips.
Eight symmetries yield 2,688 training examples; validation is not augmented.
The run uses seed 19, batch 64, AdamW at $3 times 10^(-4)$ and EMA 0.995.
Fixed-seed validation denoising loss selects update 5,000. Training takes
160 allocated seconds. This pilot does not evaluate the final-test clips.
]

#figure(
  align(center)[#table(
    columns: 2,
    align: (left,right,),
    table.header([E5 validation check], [Result],),
    table.hline(),
    [Tracked future object observations], [5,895 / 5,952],
    [Complete all-object clips; selected-object clips], [87 / 96; 93 / 96],
    [Conditional selected ADE; FDE], [0.0303; 0.0476],
    [Single-reference SSIM], [0.9517],
    [Timed contact-event F1], [0.2118],
    [Force-reversal pairs; changed-selection pairs], [14 / 16; 16 / 16],
    [Zero-force drift in first 0.5 seconds], [0 / 16],
    [Zero-force drift over full 3.875 seconds], [16 / 16],
  )]
  , caption: [One E5 training seed and four validation scenes. ADE/FDE average complete samples within cases and then scenes; missing trajectories are excluded from those conditional averages, retained in counts and still fail strict quality. LPIPS was not run for this follow-up. The selected-object probe uses shared force magnitude 0.4.]
  , kind: table
  )
<tab:trajectory>

The native-force rendered references pass only 8/16 reversal pairs in the initial window, limiting interpretation of the model's 14/16. At force 0.4, both reference and generated selection controls pass 16/16. Timed contact-event F1 has scene-bootstrap interval $[0.1316,0.2947]$\; encounter F1 of 0.8235 cannot substitute for timing. All 57 missing RGB observations occur in contact conditions. The strict tracking and selected-trajectory gate still fails.

A separate geometry audit replays every sampled pose and reproduces all saved RGB pixels exactly before scoring. Independent amodal silhouettes expose overlap hidden by renderer layer order. Raw raster overlap appears in 39/96 generated clips and 32/84 oracle-motion renders; pixel overlap alone therefore cannot diagnose penetration.

Convex hulls of initial pixel centres provide a second approximate check, with a fixed one-native-pixel tolerance. Generated hull penetration exceeds this tolerance in 22/96 clips and 369/2,976 pair frames; oracle-motion controls have zero such clips and a maximum depth of 0.855 pixels. The generated maximum is 5.737 pixels. This supports a contact failure diagnosis without changing RGB tracking or correcting model motion.

A positive control places two training or validation objects at known continuous overlaps of 0 to 4 pixels, with rotations up to 30 degrees (2,640 trials per outline offset). At the primary 0.2-pixel offset, reported hull depth has slope 1.00 and median error about $-0.25$ pixels. The one-pixel threshold flags no touching pair, about half of the 1.25-pixel overlaps, 94 percent at 1.5 pixels and every overlap of 2 pixels or more; the alternative 0.5-pixel offset flags 2 percent of touching pairs.

In the declared target groups, 15.6 percent of overlaps up to one pixel are flagged and 1.25 percent of overlaps of at least 1.5 pixels are missed. Errors have both signs, so the median is not a correction for individual maxima. Raw raster overlap exceeds one pixel for 54 percent of touching pairs. This calibrates circles and rectangles from training and validation, without triangle coverage or model-score changes.

Boundary checks remain insufficiently calibrated: 13/96 generated clips and 12/84 oracle renders exceed the one-pixel interior tolerance. Their maximum depths are 3.872 and 1.385 pixels. We retain these measurements without treating the generated count as a wall-violation accuracy result. Static-copy controls have no overlap or boundary excursions but fail nonzero-action behaviour.



=== Spread within each action family
<uncertainty-after-action-conditioning>
We stratify all 24 validation conditions by their six action families. Every family remains present, including incomplete contact ensembles. @tab:uncertainty uses selected-object trajectories. Spread is conditional on complete sample pairs; nearest-reference coverage requires all four sampled trajectories for a case. The scored-scene count is reported rather than filling missing coverage with zero.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,right,right,right,right,),
    table.header([Condition], [Model spread], [Ref. spread], [Ref. to samples], [Scenes],),
    table.hline(),
    [Forward], [0.0167], [0.0258], [0.0257], [4 / 4],
    [Reverse], [0.0248], [0.0258], [0.0171], [4 / 4],
    [Wall], [0.0440], [0.0429], [0.0272], [4 / 4],
    [Zero], [0.0147], [undefined], [0.0113], [4 / 4],
    [Contact 0 to 1], [0.0309], [0.0156], [0.0192], [2 / 4],
    [Contact 1 to 0], [0.0206], [0.0160], [0.0167], [3 / 4],
  )]
  , caption: [Whole-trajectory distances in normalized image coordinates. Model/reference spread is mean pairwise distance; reference-to-samples is the mean distance to the nearest of four sampled trajectories. Scenes denotes those with defined selected-object coverage. Zero force has one stationary reference, so reference pairwise spread is undefined. These are diagnostics, not calibrated probabilities.]
  , kind: table
  )
<tab:uncertainty>

All-object completeness is 16/16 clips for each free, zero and wall family, but only 12/16 and 11/16 for the two contact directions. All-object coverage is undefined in every scene for contact 0 to 1 and available in only one scene for contact 1 to 0. Reporting a favourable mean over the surviving contact samples would hide that limitation.

In complete timelines, estimated any-contact frequency is 0.25 under forward force and 0.1875 under zero force, versus zero in their finite simulator references. Reverse and wall conditions give zero for both. Contact-condition generated frequencies remain undefined when required timelines are incomplete. These frequencies depend on the approximate contact estimator and four samples per condition; they are not calibrated outcome probabilities.

A supplementary zero-force control verifies that all reference frames and positions remain stationary, then checks maximum excursion of any valid object over the full clip with the unchanged one-pixel motion threshold. All 16 E5 clips have complete tracks and exceed that threshold; static-reference copies drift in 0/16. The original half-second result of 0/16 drifting clips remains unchanged. Late motion explains why short-window adherence and visible sample diversity cannot establish useful hidden-physics uncertainty.

#figure(placement: auto,
  image("figures/trajectory_contact.png", width: 86%),
  caption: [
    First sorted validation contact-0-to-1 condition, one simulator reference and all four E5 seeds at fixed times. Some generated futures overlap substantially. The view retains the same image and action; visual diversity does not make every sampled motion physically valid.
  ]
)
<fig:contact>




== Contact losses and stationary objects
<contact-aware-follow-ups-and-retained-regressions>
The first two E6 attempts use the original 336 training clips and 6,000 updates each. The paired-action package adds 2,080 clips from the same 16 training worlds: both objects, eight directions, magnitudes 0.25 and 0.5, the four existing physics profiles, plus zero-force controls. Initial appearance and hidden physics remain unchanged. The original 84 validation references and all 248 final-test clips remain separate. The third attempt trains for 16,000 updates and selects update 15,500. The three TC2 A40 allocations total 24 minutes 2 seconds. Additional data and updates are confounded; this is not an isolated effect of contact loss.

#figure(
  align(center)[#table(
    columns: 5,
    align: (left,right,right,right,right,),
    table.header([Validation measure], [E5], [E6 first], [E6 token], [E6 paired],),
    table.hline(),
    [Complete all-object clips /96], [87], [94], [94], [90],
    [Full-horizon zero drift /16], [16], [0], [0], [0],
    [Full-horizon passive drift /32], [29], [20], [8], [0],
    [Penetration above one pixel /96], [22], [17], [18], [12],
    [Early reversal pairs /16], [14], [14], [16], [16],
    [Early selection pairs /16], [16], [7], [16], [16],
    [Timed contact-event F1], [0.2118], [0.3333], [0.2449], [0.3000],
  )]
  , caption: [Same 24 validation conditions and four ordered sample seeds. Zero/passive checks cover the full 3.875 seconds; reversal and selection cover the initial 0.5 seconds. Passive controls require reference-confirmed rest. One training seed per package and four validation scenes do not support an independent training-seed comparison. Every package fails strict overall quality.]
  , kind: table
  )
<tab:contact_training>

The supplementary passive-motion scorer originally counted exactly one pixel
as rest, unlike the inclusive action threshold. Version 2 rescores the same
videos with the shared rule: E5 drift rises from 22/32 to 29/32 and the first
E6 attempt from 8/32 to 20/32. Token and paired results stay at 8/32 and 0/32.
Weights, tracker and the one-pixel threshold are unchanged.

The first recipe improves contact timing but regresses target selection to 7/16. Token supervision restores the early action controls while leaving passive drift. Paired training removes observed passive drift and reduces penetration, but its timed-contact F1 is lower than the first recipe's and six clips remain incomplete. All six incomplete paired-model clips occur in wall conditions; contact conditions now track completely. This relocation of failures matters when comparing coverage. The oracle-motion control remains at 0/84 clips with penetration above one pixel. None of these results establishes exact collision laws, energy conservation or momentum conservation.



=== Spread after contact-aware training
<uncertainty-after-contact-aware-training>
@tab:contact_uncertainty retains every action family for the paired package. Forward spread is 0.0040, compared with reference spread 0.0258 and E5 spread 0.0167. Stationarity improves while free-motion samples become less dispersed than the finite reference set. Similar wall spread is insufficient: no wall scene has all four complete selected trajectories, so wall coverage is undefined. Contact-family coverage is now defined in all four scenes, with descriptive any-contact rates of 1.0 matching the finite references. Contact-event F1 of 0.3000 still shows that occurrence and timing are different checks.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left,right,right,right,right,right,),
    table.header([Condition], [Model spread], [Ref. spread], [Ref. to samples], [Scenes], [Clips],),
    table.hline(),
    [Forward], [0.0040], [0.0258], [0.0248], [4/4], [16/16],
    [Reverse], [0.0113], [0.0258], [0.0174], [4/4], [16/16],
    [Wall], [0.0409], [0.0429], [undefined], [0/4], [10/16],
    [Zero], [0.0000], [undefined], [0.0039], [4/4], [16/16],
    [Contact 0 to 1], [0.0203], [0.0156], [0.0121], [4/4], [16/16],
    [Contact 1 to 0], [0.0186], [0.0160], [0.0126], [4/4], [16/16],
  )]
  , caption: [Paired E6 package, same normalized trajectory distances as @tab:uncertainty. Scenes counts those with defined selected-object coverage; Clips counts complete all-object videos. Spread uses complete sample pairs even where coverage is undefined. Zero-force distance to continuous reference positions includes raster-centroid offset and is not evidence of video motion. Four samples per condition do not establish calibrated uncertainty.]
  , kind: table
  )
<tab:contact_uncertainty>

A saved-pose raster diagnostic explains all 32 missing observations in the six paired-model wall clips. Each is an area-change rejection: native sprites lose 50.0 to 65.4 percent of their pixels to canvas clipping, while a padded rendering of the same pose passes the existing area limits in every case. Native crops match the padded views exactly and all saved wall videos are reproduced pixel-for-pixel first. This isolates a rendering failure caused by predicted wall motion. It does not calibrate physical wall contact or replace the original failed tracks.

Saved-pose replay reproduces all 288 E6 videos, and rescoring the saved RGB
reproduces trajectory, contact and uncertainty reports. These checks verify
the reported measurements without adding training seeds or held-out scenes.

#figure(placement: auto,
  image("figures/contact_followup.png", width: 100%),
  caption: [
    E5 and paired E6 on the same first sorted validation contact-0-to-1 condition, all four seeds, at frames 0, 8, 16 and 31. Each panel includes the same single simulator reference. The case and times match @fig:contact\; no seed is selected for visual quality. Improved contact-case tracking does not imply the wall cases or all contacts are physically correct.
  ]
)
<fig:contact_followup>




== Keeping objects inside the image
<matched-training-only-canvas-penalty>
#block(breakable: false)[
The clipping diagnosis motivates a controlled loss comparison. Both recipes use the same training source, seed 19, initialization policy, balanced draws, 2,080 supplementary actions, 16,000 updates and fixed-seed validation denoising criterion. Only the canvas-loss weight changes from zero to 256. The loss squares hull excursion beyond the native image canvas after a one-pixel allowance, using the existing low-noise gate. Maximum training-label excursion is 0.638 pixels across the complete augmented corpus, below this allowance. Validation does not choose the margin. There is no inference correction, tracker relaxation or change to the Heun sampler.
]

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,right,right,),
    table.header([Validation measure], [Control], [Canvas loss],),
    table.hline(),
    [Tracked observations /5,952], [5,909], [5,951],
    [Complete all-object clips /96], [88], [95],
    [Incomplete wall clips /16], [8], [1],
    [Missing wall observations], [43], [1],
    [Penetration above one pixel /96], [15], [15],
    [Full-horizon zero drift /16], [0], [0],
    [Full-horizon passive drift /32], [0], [0],
    [Early reversal pairs /16], [16], [16],
    [Early selection pairs /16], [16], [16],
    [Timed contact-event F1], [0.3400], [0.3232],
    [Forward selected-object spread], [0.0046], [0.0042],
  )]
  , caption: [Matched L40S training runs, both selecting update 15,500, with all 24 validation conditions and four sample seeds. One training seed and four scenes give a descriptive comparison, not training-seed uncertainty. Neither model completes every all-object clip. The earlier A40 paired-model result of 90 complete clips is a separate attempt.]
  , kind: table
  )
<tab:wall_loss>

The canvas penalty improves wall visibility but leaves object-object penetration unchanged and slightly reduces timed-contact F1. Forward spread remains well below the finite references' 0.0258. Selected-object wall coverage becomes defined in three of four scenes, with reference-to-samples distance 0.0322; it remains undefined in the fourth scene. One remaining wall observation loses half its sprite area and fails the unchanged tracker. These results support a narrower rendering improvement, not validated physical wall response or calibrated uncertainty.

Both models run on the same L40S node, using 893 allocated seconds including
an initial clipping-diagnostic failure. The corrected diagnostic preserves
the native raster grid, so padded crops match the original video exactly.
It reuses the saved checkpoints without retraining. CPU replay reproduces
the 192 saved-pose videos and their reported clipping measurements.

The remaining E6 failure is sustained object overlap despite improved rest
and action control. Force extrapolation also needs separate testing: the E6
checkpoint contract limits magnitude to 1.1, while force-OOD uses 1.3 and 1.5.
Changing that limit would define a new inference experiment. E2/E3/E4 accept
1.5 and the CVAE has no explicit limit, but their training forces are still
0.15 for pushes, 0.25 for collisions and 1.1 for walls. Their force-OOD results
therefore test substantial extrapolation.



== Measuring collisions
<collision-failures-and-reference-estimator-calibration>
The saved-pose audit localizes all 15 canvas-model penetration failures to the two contact families. Every contact clip tracks completely, yet penetration exceeds the one-pixel tolerance in 142 of 2,976 pair frames. Maximum depth is 2.176 pixels, and the longest consecutive run spans 22 rendered frames. The matched control has 141 affected frames, maximum 2.216 pixels and a longest run of 21 frames. The worst example for both is scene 730016, contact from object 1 to 0, sample seed 2. In the canvas model, frames 10-31 remain above tolerance. All seeds and failures are retained; the example is selected by maximum depth. These pixel-centre hulls are approximate geometry diagnostics. Oracle-motion controls remain below tolerance in all 84 reference clips.

A separate calibration extracts contacts from every validation reference's own RGB and compares them with its own exact simulator labels. This retains the existing motion rule, thresholds and one-interval matching tolerance. All 84 reference timelines track completely, but event F1 is only 0.4889 on these clean renders; the estimator's error on generated video may differ. The matched model F1 therefore combines generated-motion errors with limitations of the contact estimator.

#figure(
  align(center)[#table(
    columns: 6,
    align: (left,right,right,right,right,right,),
    table.header([Video source], [Complete], [TP], [FP], [FN], [Event F1],),
    table.hline(),
    [Generated control], [88/96], [17], [19], [47], [0.3400],
    [Generated canvas model], [95/96], [16], [19], [48], [0.3232],
    [Own reference renders], [84/84], [22], [24], [22], [0.4889],
    [Other physics renders], [60/60], [10], [27], [38], [0.2353],
  )]
  , caption: [One-to-one timed event counts with the original fixed scorer. Generated videos use the first reference's labels. Own-reference renders use their own exact labels; other-physics renders use the first reference's labels, excluding the first reference itself. Complete counts pair timelines, including event-free cases. Undefined event ratios remain undefined; negatives do not become perfect F1. No independent-training interval is inferred from these descriptive counts.]
  , kind: table
  )
<tab:collision_calibration>

The 60 alternative hidden-physics videos are valid simulator futures for the same observed conditions. Their low first-reference F1 demonstrates additional ambiguity from hidden physical properties. It is neither a model performance bound nor an alternative score used to rank candidates. The audit retains every per-reference event comparison, including unmatched events and missing intervals, without choosing a best reference. Matching the exact simulator onsets of the same 60 alternatives with the first reference's onsets, without rendering or contact estimation, gives 9 matched, 19 extra and 39 missed events (F1 0.2368). The label-only result confirms timing variation between valid hidden-physics futures. A low single-reference timed F1 can therefore arise even without detector error.

=== Pose-based contact timing
<sec:pose-contact>
Saved rigid poses allow contact detection without the palette tracker. We move
scene outlines, used only for evaluation, by replayed simulator poses
and detect separated-to-touching transitions. Circles use 256-vertex polygon
approximations. A 0.05-pixel threshold chosen on
training clips gives validation F1 0.8462 against all 44 labelled onsets,
versus 0.4889 for pixel extraction on correct renders. Only 28 labels satisfy
the same sampled transition rule; the other 16 begin inside the touch band at
the preceding frame. The filtered 28-label result is therefore conditional
on that rule, not an independent detector test.

On the saved poses of the E7 coverage study (@sec:exploratory, @tab:coverage),
first-reference F1 is 0.3111, 0.2737 and 0.2766 for the teacher, narrow and broad arms, while valid
alternative reference poses score 0.2778. Multi-reference timing instead asks
whether a detected first contact lies within one interval of the references'
time range. Reference leave-one-out occurrence agrees in 80/80 comparisons;
four lone zero-force references have no partner. Of 32 timed comparisons,
75% lie in range, with mean nearest error 1.09 intervals.

Teacher, narrow and broad occurrence agreement is 0.9375, 0.9792 and 0.9583
over all 96 samples. Conditional on detecting contact, their 26, 30 and 28
timed samples have range agreement 0.8462, 0.8333 and 0.7143, with nearest
errors 1.00, 0.90 and 2.25 intervals. Models have four references, whereas
timed reference leave-one-out has three. These comparisons describe sensitivity
to hidden physics; eight validation contact conditions do not establish
calibration. @sec:d018-contact-timing repeats the comparison on 20 new ID
contact conditions.




== Force routing across training seeds
<independent-routing-study-and-a-separate-held-out-replication>
D016 trains six pose-space E10 models from fresh random initializations: spatial and numeric force routing at seeds 19, 20 and 21, each for 16,000 updates. The paired arms of each seed share initial tensors, batches, optimizer and contact-training recipe; routing is the declared difference. These are independent initializations, unlike E11's refinements of one E7 teacher. Both arms use future-pose/contact supervision, so this experiment addresses routing within the pose model, not the unresolved E2/E3 pixel comparison.

D017 evaluates frozen E0, both E2/E3 budgets and all six E10 models on a
separate generator seed, 790000, with ten scenes in each of five splits.
Leakage checks reject shared visible seeds or frame-zero images against the
original corpus. No checkpoint selection or training occurs on these data.
This tests repetition on a new corpus alongside the primary and longer-training
held-out results (@sec:final, @sec:d015).

#figure(
  align(center)[#table(
    columns: 5,
    align: (right,right,right,right,right,),
    table.header([Seed], [ID ADE spatial], [ID ADE numeric], [Force missing spatial], [Force missing numeric],),
    table.hline(),
    [19], [0.02604], [0.02950], [17.39], [3.77],
    [20], [0.02691], [0.02832], [12.19], [12.68],
    [21], [0.02635], [0.02992], [24.47], [5.31],
  )]
  , caption: [E10 on D017. ADE is normalized selected-object error conditional on complete trajectories within each case. Force columns are percentages of missing selected-object observations, not percentages of failed videos. All columns represent ten scene groups per arm and seed; represented groups do not establish complete tracking of every clip.]
  , kind: table
  )
<tab:routing-replication>

The spatial arm has lower conditional ID ADE in all three seeds (@tab:routing-replication); the same ordering appears on texture and repeated-physics splits with ten represented groups. Shape-OOD scores represent nine of ten groups per arm: their conditional means are descriptive, without a verified common surviving cohort. Force-OOD ADE represents only zero to five groups and is not ranked. Observation-loss rates cover all ten groups: numeric is lower in seeds 19/21, while seed 20 reverses the order. Selection also varies across seeds; for shape OOD the spatial target rates are 0.94/0.86/0.86 and numeric 0.86/0.92/0.86, conditional on complete pairs. Pair counts and missingness are retained in the export rather than interpreting a rate alone as success. E2/E3 at the original budget still lose almost all tracks on this new corpus; the larger-budget models still have selection failures. E10 and pixel models differ in representation and supervision, so their gap is not a routing ablation.

A later compact-archive audit recovers 1,080 E10 per-case trajectory reports
for ID, shape and texture. Complete all-object clips per 240 samples range
from 230-240 on ID, 211-216 on shape and 229-240 on texture. Raw videos were
deleted from TC2, and the force/physics and earlier-model reports are absent.
The compact counts can be checked, but the RGB collection cannot be replayed
independently. The conditional ADE ordering also compares arms with different
missing observations; it is not an unconditional accuracy ranking.

= Physical structure and sampled worlds
<sec:exploratory>
The next studies examine why the pose models fail. Initial E7/E8/E9/E11
comparisons reuse the original four validation scenes and usually one training
seed. Their pose and physical-property supervision exceeds the main RGB-only
comparison. D018 later tests fixed-teacher refinements on ten new ID scenes;
the geometric and dynamics studies use their own stated corpora. These
experiments do not replace the primary held-out results.

E7 adds learned relational denoising and a mass/friction/restitution latent to the E6 wall backbone. Physical labels supervise the latent during training. At inference it draws from the simulator's declared prior, keeps that draw fixed throughout denoising and reuses it across interventions with the same seed. It does not recover the supplied scene's hidden physics from a resting image. The conditioned arm completes 96/96 all-object clips, versus 95/96 for its E6 teacher, and reduces penetrating clips from 15 to 11. Maximum depth nevertheless increases from 2.1762 to 2.4480 pixels. Timed contact F1 remains 0.3366 and ten of 24 trajectory conditions fail. Better visibility therefore does not establish correct contacts or calibrated uncertainty.

E8 freezes the E7 teacher and trains a temporal residual, comparing reset versus carried recurrent memory at a matched parameter count and update budget. Both retain 96/96 complete clips and 11/96 penetrating clips, with maximum depths of 2.4516 and 2.4502 pixels respectively. The residual changes the motion slightly but does not resolve contact failures. These results do not support a benefit from carrying this memory.

A paired coverage study then warm-starts the same E7 teacher in both arms, trains every parameter for 8,000 updates, and adds 4,160 clips per arm. Narrow reuses old visible scenes; broad introduces independent visible scenes. Each resulting cache has 52,608 augmented examples, with identical sampling weights, initialization and private random state. Actual extra-clip contact prevalence differs (1,376 narrow versus 1,134 broad), so this changes the visible distribution rather than isolating geometry count.

#figure(
  align(center)[#table(
    columns: 4,
    align: (left,right,right,right,),
    table.header([Measure], [E7 teacher], [Narrow], [Broad],),
    table.hline(),
    [Complete all-object clips /96], [96], [96], [95],
    [Penetrating clips /96], [11], [9], [15],
    [Penetrating future frames], [105], [99], [176],
    [Maximum depth (pixels)], [2.4480], [2.4895], [2.7101],
    [Timed contact F1], [0.3366], [0.3366], [0.3529],
    [Failed trajectory conditions /24], [10], [10], [10],
    [Correct full-horizon targets /64], [62], [61], [58],
    [Passive drifts in those pairs /64], [2], [3], [6],
  )]
  , caption: [Exploratory validation coverage study, one training seed and four reused scenes. Full-horizon selection excludes eight of 24 conditions whose references contact first. Full-horizon force scaling completes 75/76 eligible comparisons in each refined arm, increasing displacement in 75 narrow and 74 broad comparisons; the missing comparison is retained. Thirteen force conditions are excluded because their references contact first.]
  , kind: table
  )
<tab:coverage>

The coverage comparison uses 1,708 allocated A40 seconds. Independent replay
reproduces all 288 saved clips; geometric depths differ by less than
$2.79 times 10^(-7)$ pixels across platforms. Neither arm uses an inference
solver, motion projection or candidate selection.

E9 compares local vertex-to-segment messages with a parameter-matched pooled
control. Each arm refines the same E7 teacher for 8,000 updates on its original
19,328-example cache, using 1,079,944 parameters and matched random state.
Both complete 96/96 clips. Local and pooled yield 12 and 11 penetrating clips,
maximum depths 2.5035 and 2.5176 pixels, and contact F1 0.3600 and 0.3673.
Both retain ten failed trajectory conditions, so local messages show no
contact-quality advantage.

Both arms achieve 64/64 full-horizon target pairs without passive drift.
Force scaling completes 75/76 local and 76/76 pooled comparisons, with motion
increasing in every complete comparison. The native E7 anchor gives 63/64
target and 76/76 force pairs, compared with 62/64 and 75/76 in the earlier
Windows CPU probe. This small platform-sensitive difference is not a training
gain. Independent replay reproduces all 288 clips, including an anchor
identical to the earlier coverage study. The run uses 1,819 A40 seconds.

== One-pass decoding versus diffusion
<noise-after-conditioning-on-a-sampled-world>
E11 compares a one-pass pose decoder with conditional diffusion after sampling the same physical prior. Both refine the same E7 tensors on the original cache for 8,000 updates at seed 19, with identical batches, random draws, loss weights, optimizer and nominal 1,064,580 parameters. The direct arm replaces noisy motion with zero and fixes the decoder noise level to one. Its motion-input weights have zero gradient, so active capacity differs; the transferred function is preserved only in the diffusion control. Hidden subpixel geometry remains unknown even after conditioning on mass, friction and restitution.

#figure(
  align(center)[#table(
    columns: 3,
    align: (left,right,right,),
    table.header([Measure], [One pass], [Conditional diffusion],),
    table.hline(),
    [Network evaluations per video], [1], [31],
    [Complete all-object clips /96], [96], [96],
    [Penetrating clips /96], [1], [11],
    [Penetrating future frames], [4], [105],
    [Maximum depth (pixels)], [1.0557], [2.5040],
    [Longest penetration (frames)], [4], [19],
    [Timed contact F1], [0.3107], [0.3711],
    [Zero-force drifting clips /16], [2], [0],
    [Mean selected-object ADE], [0.02514], [0.02466],
    [Failed trajectory conditions /24], [7], [10],
    [Forward trajectory spread], [0.02498], [0.02346],
    [96-video sampling/export (seconds)], [2.31], [31.06],
  )]
  , caption: [Exploratory E11, one training seed and four reused validation scenes. ADE and spread use normalized image coordinates. Sampling includes native rendering/export but excludes checkpoint loading; it is not browser latency. The finite reference forward spread is 0.02580.]
  , kind: table
  )
<tab:prior>

One-pass decoding reduces penetration and sampling cost but worsens rest,
contact timing and selected-object ADE (@tab:prior). Both arms achieve 64/64
full-horizon target pairs without passive drift. Force scaling completes
76/76 direct and 75/76 diffusion comparisons, with displacement increasing in
each complete comparison. Eight selection and thirteen force conditions are
excluded because their references reach contact. All 288 primary clips replay
exactly. The run uses 1,497 allocated A40 seconds. The tradeoff is measured on
one refinement seed and four reused validation scenes; D018 tests its repeatability.

== Repeating the decoder comparison on new scenes
<fresh-scene-replication-of-the-prior-decoder>
D018 fixes six final-update checkpoints before generating a new corpus at
seed 830000. Seeds 20/21 repeat the two 8,000-update refinements; seed 19
comes from the earlier screen. These seeds share one E7 teacher rather than
training six systems from scratch. All arms use the original training cache
and losses. Leakage checks reject frame-zero or visible-seed overlap against
all 668 original clips. Evaluation uses ten new ID scenes: 60 conditions with
four samples each, plus 80 selection-control videos per model. Other generated
splits are validated but receive no model scores.

#figure(
  align(center)[#table(
    columns: 8,
    align: (right,left,right,right,right,right,right,right,),
    table.header([Seed], [Decoder], [Complete], [Pen.], [Max. px/frames], [Rest drift], [F1], [ADE],),
    table.hline(),
    [19], [One pass], [240/240], [5], [1.578/6], [5/40], [0.2745], [0.03076],
    [19], [Diffusion], [240/240], [26], [2.791/24], [0/40], [0.3053], [0.02918],
    [20], [One pass], [240/240], [10], [1.553/12], [5/40], [0.3015], [0.03021],
    [20], [Diffusion], [240/240], [24], [2.635/23], [0/40], [0.3191], [0.02926],
    [21], [One pass], [240/240], [2], [1.100/6], [0/40], [0.3005], [0.02980],
    [21], [Diffusion], [240/240], [22], [2.584/23], [0/40], [0.3118], [0.02947],
  )]
  , caption: [D018 on ten fresh ID scenes. Penetration counts use the unchanged one-pixel threshold. Max. px/frames gives worst depth and longest consecutive penetration. Rest drift is zero-force full-horizon drift; every zero-force sample is tracked. F1 uses timed contacts against the first declared reference; ADE is normalized selected-object error, defined for all 60 conditions here.]
  , kind: table
  )
<tab:prior-replication>

All six models track every primary clip. One-pass reduces penetration, worst depth and duration in each refinement seed, but every seed fails the frozen overall improvement rule. All three regress selected ADE, timed contact F1, early force-scaling success, wall coverage and object-1-to-0 collision-family coverage. Seeds 19 and 20 also introduce five zero-force drifts out of 40; seed 21 introduces none. Coverage is checked separately for all six action families, with all ten scene groups required; a pooled gain cannot offset a regression. The 0.5-second response/reversal checks were also present in the committed decision code before the new data was generated.

#figure(placement: none,
  image("figures/prior_replication.png", width: 100%),
  caption: [
    First sorted D018 ID condition, refinement seed 19. Every sampling seed and all four reference variants are shown at the same four times. Reference row indices do not identify the models' sampled physical worlds. This is a fixed-case visualization, not a selected best sample or a calibrated posterior.
  ]
)
<fig:prior-replication>


=== First-contact timing with multiple references
<sec:d018-contact-timing>
After every D018 decision was frozen and scored,
we applied the pose-based first-contact measure of @sec:pose-contact
to the saved D018 poses, using the touch threshold chosen on the original training
clips. Twenty of the 60 conditions involve contact. Reference occurrence uses 200 scorable leave-one-out comparisons; ten lone zero-force references have no comparator. Scored against the other
references of their condition, the simulator references place 77.5% of first
contacts within one interval of the others' range, with mean nearest error 1.24
intervals. The diffusion arms reach 0.7460, 0.7333 and 0.7288 for seeds 19, 20 and 21,
with mean errors 1.65, 1.65 and 1.54 intervals, conditional on 63, 60 and 59 scorable contact-time comparisons among 240 samples per arm. The one-pass arms reach only 0.5581,
0.6111 and 0.5532, with mean errors 3.07, 2.83 and 2.83 intervals on 43, 54 and 47 timed samples, and miss contact
more often (occurrence agreement 0.8458-0.8917 against 0.9125-0.9292). The ordering
holds in every seed and at outline offsets of 0, 0.2 and 0.5 pixels. This supports
the timed-contact regression in @tab:prior-replication with a measure that
uses all references and no pixel contact estimation. It is a descriptive held-out
check and changes no gate or selection.

D018 inference runs on an RTX 3060 with deterministic algorithms and TF32
disabled. All 1,440 saved primary videos replay exactly from their poses.
Concurrent CPU checks prevent an isolated timing comparison, so the earlier
controlled E11 screen supplies the cost comparison. The reduction in penetration
repeats across refinement seeds, but contact timing and action control worsen,
so we cannot call the one-pass decoder an overall improvement.

== Contact projection and simulator controls

<sec:contact-projection>

D028 tests an explicit geometric layer after sampling the D020 neural pose
trajectory. D020 subtracts the backbone's zero-force response from its action
response for each sampled world. It uses a seed-19 checkpoint trained for
8,000 updates from the frozen E7 teacher/cache. Inspired by position-based dynamics @pbd and physics-guided
motion projection @physdiff, it separates overlapping observed pixel
hulls using inverse sampled-mass weights and keeps hulls within the canvas.
Our layer runs after sampling, preserves the neural weights and rotations,
and changes only violating positions. It is a geometric hybrid, rather than
evidence that the neural model has learned contact dynamics.

The mode is selected on original validation. Its rules and non-inferiority
margins are fixed before generating ten fresh ID scene groups, with frame-zero
overlap against opened corpora rejected. Both arms use one frozen checkpoint
and identical prior draws. All 240 clips per arm are tracked completely.


#figure(
  table(
    columns: 3,
    align: (left, right, right),
    table.header([Fresh D028 ID], [Neural D020], [With projection]),
    table.hline(),
    [Penetrating clips, over 1 px], [19/240], [0/240],
    [Boundary-clipped clips], [25/240], [0/240],
    [Rest / passive drift], [0/40; 0/80], [0/40; 0/80],
    [RGB contact F1], [0.169], [0.176],
    [First-reference selected ADE], [0.02495], [0.02476],
    [Same-world selected ADE], [0.00706], [0.00712],
    [Same-world other-object ADE], [0.00320], [0.00355],
  ),
  kind: table, caption: [D028 contact projection, with the unchanged scorers. ADE uses normalized
coordinates. Geometry improves, while other-object same-world error rises.],
)
<tab:projection>


The paired scene-bootstrap 95% interval for selected same-world ADE change
is $[-0.00014,0.00028]$, below the prespecified upper margin of 0.0005.
The projection passes its registered rules. Other-object error nevertheless
rises by 11%; that regression was not an acceptance rule and remains reported.
The unchanged strict quality flag is false for both arms. This is one checkpoint
on ten ID groups, with no OOD or calibrated physics claim.

After D028 was frozen, we applied the pose-based first-contact measure of
@sec:pose-contact to the saved D028 poses, with the touch threshold
chosen on the original training clips. Twenty of the 60 conditions involve
contact. The projected samples agree with the references on whether contact
occurs in 0.7958 of samples, against 0.9042 for the raw sampler: 49 of 80
projected contact-condition samples never reach exact contact, against 23 raw.
The projection separates observed pixel-centre hulls. At true simulator contact,
these hulls overlap in 491 of 1,032 reference frames (25th percentile $-0.70$
pixels), so the projection stops such objects before their exact outlines meet.
In 35 of the 49 projected misses, the hulls touch at the closest frame while the
outlines remain a median 0.56 pixels apart. In the other 14 misses, the hulls
are still apart, so the sampled motion itself stops short.

With a one-pixel touch
tolerance, used only as a sensitivity check, both arms reach contact in every
contact condition, and the projection's mean first-contact error of 0.70
intervals is close to the raw 0.60 and the references' 0.57. At 0.5 pixels,
however, its error is 3.11 intervals against 1.78 raw. The projection thus
leaves a sub-pixel standoff that delays or removes exact first contact, and the
RGB contact score cannot detect it.

D030 re-renders the simulator with true hidden physics and with the four prior
draws. On two fresh corpora, one of them D028's, the true-physics RGB contact F1 is
only 0.590 and 0.250, exposing the estimator's limitations. Sampled-physics ADE is close
to the model results, consistent with a substantial hidden-physics contribution.
These are descriptive reference controls, not mathematical method ceilings.

D029 compares an 8,000-update, seed-19 direct decoder with the same recipe plus
contact-transition and signed contact-depth-change losses, weighted 8 and 4.
On four original validation scenes, both arms complete all 96 clips, retain
1/96 penetrating clips and 2/16 zero-force drift, and score RGB contact F1
0.3048. Selected ADE is 0.025146 for the control and 0.025156 for the candidate.
This single-seed result shows little change; @sec:distribution-followups
examines the scale of the added losses.




== Distributional evaluation and loss scaling

<sec:distribution-followups>

The original acceptance rule compares sampled motion with one realized
reference. Simulator controls show that it can reject a valid hidden-physics
alternative. D031 instead compares a forecast ensemble with all available
references using the fair energy score. For $K$ trajectories and reference $y$,
$hat("ES")(X,y)=1/K sum_(i=1)^K d(x_i,y)-1/(2 K (K-1)) sum_(i != j) d(x_i,x_j)$,
where $d$ is the Euclidean trajectory distance divided by the square root of
the number of frames. We score displacements in native pixels and average
over references. The underlying score is proper @gneiting; the distinct-pair
form removes finite-ensemble bias under independent sampling @ferro.
Four samples and at most four references still give noisy condition estimates,
so comparisons use paired scene-group intervals.

On D028, D020 scores 1.105 pixels against 1.060 for a simulator using the same
physical prior; the paired difference interval is $[0.019,0.074]$. A baseline
with a 0.15-pixel hull offset lies within the simulator's sampling interval on
validation, D028 and D033. Every learned pose checkpoint included in this
comparison scores worse than the simulator. Matching single-reference ADE
does not imply a matched forecast distribution.

D032 checks the scale of losses in matched retraining. The D020 recipe's
weighted geometry terms contribute about 0.5% of total training loss; D029's
contact-dynamics terms contribute about 0.02%. Geometry weights are
multiplied by 16, 128 or 1024; contact-dynamics weights by 128 or 1024.
Original-validation selection chooses rest-anchored geometry at 16 times its
weight and direct contact dynamics at 1024 times. Every tested geometry scale
reduces penetration but worsens same-world ADE and energy score. Retrained
controls reproduce the original weights. Direct models are allowed to drift
under zero force; their actual drift is scored rather than assumed to be zero.

D033 evaluates the selections and six E10 checkpoints on ten new ID scene
groups. The projection passes its declared checks for all ten checkpoints.
The direct contact-dynamics selection improves at training seed 19, but D034
fails to replicate that complete rule at either seed 20 or 21: wall overshoot
and lost tracks remain. Penetration alone reaches zero at all three seeds.
The raw losses do not establish a general geometry improvement.

D035 tests the scaled direct losses together with projection on another ten
new ID groups, generated after that comparison was registered. All three
training seeds pass the combined rule. Same-world selected ADE falls by
8 to 10%, or 0.03 to 0.05 native pixels, with zero projected geometry violations
or lost tracks. The comparison was chosen after seeing D034; the failed raw
replication is retained. Selected first-reference ADE and energy score do not
improve at every seed, and every learned run remains worse than the simulator
on energy score. This supports the limited hybrid comparison, not an overall
learned-physics or OOD claim.

== Selecting four futures from sixteen

<sec:candidate-selection>

D036 generates 16 candidates from one frozen E11 checkpoint on the already
opened D018 ID corpus, then selects four using visible hull violations,
stationarity, force opposition and trajectory diversity. The selector sees no
reference future or true hidden physics. Its fair control is the exact
expectation over uniform four-of-16 subsets, rather than a favourable or
unfavourable choice of the first four seeds.


#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header([D036, 240 returned clips], [Uniform expectation], [Selector], [Oracle]),
    table.hline(),
    [Selected ADE], [0.02621], [0.02760], [0.01245],
    [Complete clips], [237.25], [240], [240],
    [Penetrating clips], [19.0], [0], [16],
    [Boundary-clipped clips], [27.75], [14], [29],
  ),
  kind: table, caption: [D036 candidate selection. Uniform ADE averages complete candidates;
its missing-clip expectation is reported separately. The oracle uses the
reference future and is unavailable at inference. Geometry counts use the
same hull definitions as the selector and are not independent evidence.],
)
<tab:selection-negative>


Selection raises completeness but worsens ADE by 5.28% against the fair
control. The oracle identifies substantial candidate-pool headroom that this
hand-written score cannot recover. Sixteen candidates also require four times
the model sampling of four direct outputs. The selector is not promoted to
the public sampler. The accepted D028 projection is exposed separately as an
explicit optional correction, preserving the ordinary seeded sampling path.

== Dynamics residual and momentum correction

<sec:dynamics-residual>

D038 adds two separate optional components. A training residual matches
finite-difference free-motion acceleration to the known force divided by
sampled mass, excluding object and wall contacts where impulses make that
target incomplete. An inference block corrects post-force linear momentum
using the known applied impulse and sampled masses. Before predicted contact,
the correction acts on the selected object; after contact, it adjusts the
centre-of-mass motion while preserving relative velocity. Rotation is
unchanged. Trajectories approaching a wall are skipped because wall impulse
is unobserved. The inference order is neural sampling, momentum correction,
then the accepted hull projection. These are explicitly labelled hybrid
corrections, rather than evidence that the model learned the enforced laws.

The matched seed-20 pair uses the same E7 teacher, initialization, private RNG
and 8,000 updates; only the dynamics-residual weight differs, zero versus four.
The fresh seed-980000 corpus contains ten ID scene groups, 60 conditions and
240 generated clips per arm. All 355 stored simulator clips replay exactly.
The rule was committed locally before generation, but the registration and
results first reached GitHub together. Remote history therefore does not
independently establish the registration ordering.


#figure(
  table(
    columns: 3,
    align: (left, right, right),
    table.header([D038 fresh ID], [Projected control], [Full hybrid]),
    table.hline(),
    [Complete clips], [240/240], [240/240],
    [Penetrating / boundary clips], [0 / 0], [0 / 0],
    [Zero-force / passive drift], [0 / 0], [0 / 0],
    [Same-world selected ADE], [0.00699], [0.00582],
    [First-reference pose ADE], [0.02771], [0.02665],
    [Free-motion dynamics residual], [0.00564], [0.00101],
    [Energy score, pixels], [1.2605], [1.2539],
    [RGB contact F1], [0.228], [0.213],
  ),
  kind: table, caption: [D038 full hybrid: dynamics-residual training, momentum correction and
projection. ADE uses normalized coordinates. First-reference pose ADE is
distinct from RGB-tracked ADE. Zero geometry violations are measured on
observed hulls and do not establish complete physical validity.],
)
<tab:dynamics-residual>


Same-world ADE falls by 16.8%, with a paired scene interval
$[-0.00149,-0.00084]$; first-reference pose ADE falls by 3.8%. Same-world ADE
compares the sampled trajectory with a simulator using the same sampled
physical coefficients. Because the momentum block also uses those masses,
the component comparison in @tab:dynamics-components is needed to separate
the learned residual from the explicit correction. Within the block's wall-free scope, momentum is set by construction; it is
not a learned conservation result.


#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header([D038 component], [Same-world selected ADE]),
    table.hline(),
    [Control plus projection], [0.00699],
    [Add momentum correction], [0.00618],
    [Residual training plus projection, no momentum block], [0.00688],
    [Residual plus momentum plus projection], [0.00582],
  ),
  kind: table, caption: [D038 component attribution. The explicit momentum block accounts for
about two thirds of the aggregate gain and all of the wall-family gain.],
)
<tab:dynamics-components>


The full pipeline's collision-family ADE rises by 1.17%, and RGB contact F1
falls. The momentum block alone worsens collision motion; the residual
recovers most of that loss. Without the block, residual training reduces its
trained dynamics error by 80.7% and raises contact F1 from 0.228 to 0.278,
but gives only a small same-world ADE improvement. One training
seed and ten ID groups support no collision-specific or OOD claim. An independent
A40 retraining reproduces the principal values to about four significant figures
on a regenerated corpus; this is a
cross-platform reproduction, not another training-seed replication.

D039 retains a separate exploratory collision-impulse screen. Both seed-22
arms train for 2,000 updates with the dynamics residual and use momentum
correction plus projection; only the collision-residual weight differs, zero
versus 1024. On the already opened D038 corpus, collision same-world ADE
changes from 0.00683 to 0.00606, with interval $[-0.00117,-0.00043]$, while
contact F1 changes from 0.168 to 0.238. Both retain 240 complete clips without
measured geometry violations or drift. First-reference pose ADE becomes
slightly worse, and raw neural collision ADE worsens by 3.2%. A shortened,
single-seed screen on an opened corpus is not confirmation or evidence of a
standalone learned-collision improvement. The default-zero loss remains
experimental.

= Interactive comparison
<sec:demo>
The interactive interface compares the flow CVAE, E5 and E6 on a common image
with four fixed sample seeds. Users select an object, drag a force and compare
an action with its reversal on synchronized timelines. Simulator reference
worlds are labelled separately and do not affect model outputs. Equal seeds
pair random draws; they do not identify the true physical world. An optional
E7 profile reuses its sampled prior properties across pushes. Native 64px
synthetic uploads derive object masks from pixels and return four 32-frame
futures with the exact input frame prepended. Uploaded images have no known
simulator reference. This interface visualizes the measured models; it is not
an additional accuracy experiment.

== Force Prompting photograph generation
<sec:photo>
We implement a separate photograph interface using the released Force Prompting point-force checkpoint and CogVideoX-5b-I2V~@forceprompting. This model was trained externally. Its outputs are excluded from every controlled-model score above. The interface centre-crops an uploaded RGB image, records a point and force vector, and retains four declared sampling seeds. A direction-independent caption describes the scene and the action. Pinning one action permits synchronized comparison against a reversed push. Its force scale and sample variation have no physical calibration.

The full released transformer and ControlNet run on one A40 without layer truncation. Native output contains 49 frames at $720 times 480$ and 8 fps. A preregistered development comparison (PHOTO-P02) keeps the image, caption, action and seeds 0, 1, 2 and 3 fixed while comparing 20 and 50 denoising steps. Median generation times are 192.84 and 466.42 seconds per sample, respectively. Motion is weak in both profiles. An independent repeat of the 20-step seed-0 output is byte-exact at the raw RGB level. We select 20 steps for engineering cost, without claiming equivalent general quality or improved control.

PHOTO-P03 uses a public StreamForce development image of a mug on a desk
@streamforce. Two fresh requests keep the image, neutral caption, point,
force magnitude and seeds fixed while reversing force direction. Each returns
four native clips through the browser and allocated GPU worker, taking 788.88
and 792.06 seconds. These times include the end-to-end request path.

All four reversed-push samples show leftward mug translation. In the right-push set, seeds 0 and 1 respond weakly or remain nearly static, whereas seeds 2 and 3 translate right (@fig:photo-pair). The mug and room remain recognizable, with shape, hand and background variation. This manual inspection concerns one reused public development image. It establishes neither physical reliability nor generalization, and matched noise seeds do not identify a shared hidden world. The native video model also reconstructs its initial frame, unlike the synthetic interface's exact input-frame copy.

The interface supports cancellation before or after a completed sample,
retains partial output and labels saved playback as recorded replay.
Offline requests that have no saved result are rejected. Recorded replay
works without a VPN or GPU; fresh generation requires an allocated worker.



#figure(placement: none,
  image("figures/photo_paired_native.png", width: 100%),
  caption: [
    PHOTO-P03, every declared sample for both pushes. Columns show generated frame zero and 6.0 seconds. Each crop uses the same input-defined horizontal band, rows 350 to 479 of the native 480-row output; no generated pixels are altered. Weak right-push samples remain included. External model: Force Prompting/CogVideoX. Input: StreamForce public development gallery.
  ]
)
<fig:photo-pair>




== StreamForce photograph generation

<sec:photo-streamforce>

The released StreamForce student EMA @streamforce provides a second,
separately attributed photograph backend. PHOTO-S01 checks all four native
seeds and a seed-0 repeat on one A40. Each clip has 81 frames at $832 times 480$
and 16 fps, using the released four-step sampler. Every float output is finite
before conversion to RGB. Generation takes 20.50 to 22.22 seconds per clip in
this smoke run after loading. The repeated seed matches exactly. The official
smoke caption already describes motion, so this result alone cannot establish
force control. These outputs do not enter the controlled-model scores.

PHOTO-S02 uses the same public mug image, neutral caption, point, magnitude
and seeds for opposite pushes. Both fresh requests run through the browser,
SSH queue and allocated worker. They take 132.73 and 134.28 seconds, including
a fresh model load for each action and presentation-file transfer. Every raw
RGB array and MP4 matches its local backup digest. Fixed-frame manual review
shows rightward mug translation in all four right-push samples and leftward
translation in all four reversed samples (@fig:photo-streamforce).
Mug size and shape, hands and the background still vary. One reused public
development image provides no physical ground truth, generalization estimate
or calibration. The different native crop, horizon and sampler also prevent
an isolated quality or speed comparison with PHOTO-P03.

StreamForce uses the same queue and replay interface. Cancellation preserves
completed samples, and a subsequent four-seed request verifies recovery.
Both primary pushes can be replayed offline with all eight videos. These
checks concern interface operation, not physical accuracy.





#figure(placement: none,
  image("figures/photo_streamforce_native.png", width: 100%), caption: [PHOTO-S02, every declared native StreamForce sample for both pushes.
Columns show frames zero and 80 (5.0 seconds). All rows use the same
input-defined horizontal band, rows 320 to 479 of the 480-row output, without
altering generated pixels. External model: released StreamForce student EMA.
Input: its public development gallery, point\_1. These are manually inspected
development outputs without calibrated physical ground truth.])
<fig:photo-streamforce>





= Conclusion
Click2Future provides a controlled path from an image and force to four
generated futures, with replayable data, trained baselines and synchronized
comparison. The flow CVAE preserves appearance and follows push direction
over the full clip at the training force. Its slower motion and wrong-object responses show why
appearance scores need separate force and selection tests. Longer pixel-diffusion
training improves image quality, but object-identity errors leave the numeric
versus spatial comparison unresolved for these models.

The pose studies improve tracking, object identity and action response.
Contact-aware training reduces unforced drift and overlap; the matched canvas
loss improves wall visibility. One-pass decoding lowers sampling cost and
reduces penetration across all three D018 refinement seeds, while worsening
trajectory error and contact timing. Explicit projection removes measured hull
violations, but exact-outline scoring reveals that it can stop objects short
of contact. These controls separate visibility and geometric gains from
accurate contact dynamics.

The D038 hybrid reduces same-world ADE by 16.8% on ten fresh ID scenes.
Component comparisons attribute most of that gain to explicit momentum
correction, while collision error and RGB contact F1 regress. Multi-reference
energy scores still favour the prior-matched simulator. The photograph extension
adds force-controlled generation with separately attributed external models;
its development examples establish interface operation, not generalization.

The next experiment should vary free-push magnitude and selected object
independently during training, then repeat the force and selection tests on
new scenes. This would test the coverage explanation directly. For contact,
we need to compare exact-outline timing and multi-reference scores alongside
hull overlap: the projection study shows that removing overlap can also remove
contact. Any gain should survive independent training seeds, with missing
tracks reported beside conditional trajectory errors.

#bibliography("references.bib", style: "ieee", title: "References")

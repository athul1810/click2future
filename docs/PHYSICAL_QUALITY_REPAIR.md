# D019: investigate and stabilize direct-decoder training

The demo software runs, but physical quality is a separate question. D018 tracks
all primary clips while the one-pass model penetrates in 5, 10 and 2 of 240 clips
and adds zero-force drift in 5 of 40 samples for two refinement seeds. Every seed
regresses in contact timing, selected trajectory error, early force scaling and
two coverage families. Those results remain unchanged.

## Reproduced training issue

The E11 direct decoder ignores corruption and sigma in its forward pass. The
inherited loss still multiplies regression, stationary and acceleration errors
by `16 + 1/sigma^2` and randomly enables geometry losses when `sigma <= 1`.
For an identical real checkpoint prediction and zero-force validation example,
the output-weight gradient norm changes from 0.353 at sigma 1 to 4,998 at sigma
0.002. In 100,000 reproducible draws, the highest-weight one percent carry 49.8%
of the total weight. This establishes unnecessary objective variance. It does
not establish that variance is the only cause of the video failures.

Other limits remain: training uses sixteen independent visible geometries, the
pose decoder has no hard conservation/contact constraints, and 64px observations
hide subpixel geometry. Reference contact timing also varies with hidden physics.
Increasing the count of actions on existing scenes does not add new geometries.

## Fixed paired experiment

Use the original frozen cache (19,328 augmented examples) and E7 teacher, with
seed 19, batch 64 and 8,000 final updates. Both arms are direct decoders with
identical initial tensors, batches, optimizer, schedule, loss weights, renderer
and inference. Recipes differ only in `expected_objective`. Both save every
250 updates. No model output is projected, filtered or overwritten.

The control uses the original stochastic direct loss. The candidate computes
its exact expectation over the clipped lognormal sigma distribution. Point,
temporal, acceleration and rest terms retain their expected EDM coefficient;
geometry terms retain their original activation probability. Neither the
average objective nor physical scoring thresholds are intentionally changed.
The candidate still consumes the same private RNG draws to keep later batches
paired. Nonlinear tail-risk losses are excluded because that reduction does
not apply to them.

Checkpoint format 8 distinguishes this experiment from frozen E11 format 7.
Its architecture and inference are unchanged. Format-8 candidates may be
evaluated only on original validation and derived selection controls. D017,
D018 and the old test splits cannot be reused to tune or promote this change.

Evaluate both final checkpoints with the existing full validation suite,
amodal geometry, zero/passive drift, contact audit, both action horizons,
selected-object controls and action-family coverage. Retain every sample and
failed phase. A lower training loss is insufficient: report all physical
regressions. A validation improvement requires independent repeats and a
separately frozen new holdout before a general quality claim. No model has yet
been promoted by D019.

## VPN and interruption recovery

Run `cluster/submit_once.py` detached on TC2 with a unique experiment job name.
It locks submission, atomically saves a job receipt, returns the same receipt
on retry and can recover an active allocation after a lost acknowledgement.
Do not delete a receipt or resubmit because a local SSH connection timed out.
Read the receipt and query Slurm first. For a terminal failed job, inspect its
logs before starting an explicit new attempt with the same immutable inputs.

Slurm training runs independently of the VPN. The trainer atomically saves a
full model/EMA/optimizer/private-RNG checkpoint, including at initialization.
The driver forwards allocation stop signals so training can checkpoint. Resume
requires exactly matching source, cache, teacher and configuration. Completed
arms are hash-checked and reused; evaluation retries use new directories and
retain incomplete reports. Interrupted phases exit unsuccessfully and cannot
be mistaken for completed quality checks.

Use resumable `rsync --partial --checksum` transfers. Validate SHA-256 after
every completed upload/download before extraction or analysis. Keep copies of
the source, input identities, job receipt, logs and results outside TC2 because
the course cluster provides no backups.

The scheduled comparison uses one TC2 GPU, four CPUs, 24 GB system RAM and a
one-hour allocation. Each training arm has a twenty-minute safety bound; a
time-limited arm retains its checkpoint and requires explicit resume. Actual
cost and outcome will be added after the job. No additional task deadline is set.

# TC2 access status

Access is approved for BOBAN ATHUL KRISHNA (`athulkri001`) and MAITI SHRIJEET
(`shrijeet001`). Shrijeet's approval was supplied on 16 October 2026. Both approvals
list partition `MGPU-TC2`, QoS `normal`, and the limits in [COMPUTE.md](COMPUTE.md).

On 16 October, Shrijeet's Windows preflight confirmed an active GlobalProtect adapter,
an NTU route covering the TC2 address, and successful TCP port 22 connectivity.
The initial SSH attempt closed, and Shrijeet confirmed a successful plain interactive
SSH login on 16 October; the terminal confirmed home-directory creation. Setup job
`42886`, using source commit `47ca5e0`, completed on `TC2N01` with exit code `0:0`.
The A40, pinned environment, dependency check, CUDA gradient/checkpoint check, test
suite and 31-frame synthetic profile passed. See the
[sanitized allocation evidence](results/tc2_probe_2026-10-16.json).
Private connection reports stay under ignored
`outputs/tc2-setup/`; they are not part of the repository.

Follow the [TC2 setup workflow](../cluster/README.md) for first login, committed-source
upload, hardware inventory, isolated dependency installation, and the GPU probe.

Real training jobs 42909/42910 also completed. Full model, EMA, optimizer,
scheduler, RNG and data-order continuation matched the next update across
allocations with zero tensor difference. Local backups include the full checkpoint
and result bundle. The first model failed generated-video quality checks; that is
a learning issue rather than an access failure. See
[training evidence](results/diffusion_pilot_2026-10-16.json).

Storage quota, total usage allowance and access expiry remain unconfirmed.

17 october: vpn reachability and authenticated ssh worked after reconnecting.
jobs 43389/43396 prepared the frozen corpus and trained the trajectory pilot on
the a40; job 43404 completed validation after installing pinned scikit-image
0.26.0. dependency checks and exact cuda checkpoint continuation pass. the first
evaluation attempt, 43403, failed on that missing dependency and is retained in
[the experiment record](results/trajectory-pilot-validation.json). weights and
results have verified local backups. access is working; model contact quality
remains open. no per-user storage quota or access expiry was exposed by these checks.

Access for the remaining members is not recorded. Members who need their own account should
follow the course application form and send it to `ccdsgpu-tc@ntu.edu.sg`. Keep completed
forms, credentials, and private connection reports outside Git.

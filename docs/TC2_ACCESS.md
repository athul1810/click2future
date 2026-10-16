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

Storage quota, total usage allowance and access expiry remain unconfirmed. Full
training-state resume across jobs belongs to the trainer work, before long runs.

Access for the remaining members is not recorded. Members who need their own account should
follow the course application form and send it to `ccdsgpu-tc@ntu.edu.sg`. Keep completed
forms, credentials, and private connection reports outside Git.

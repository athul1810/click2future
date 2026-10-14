# TC2 access, installation, and first GPU job

Access is approved for Athul with partition `MGPU-TC2` and QoS `normal`.
See [the account limits](../docs/COMPUTE.md). First login, Linux package installation,
and allocated GPU execution still need verification. The scripts prepare those checks.

The scripts were adapted from `CCDSGPU-TC2-Masters-UserGuide.pdf`, especially pages 4,
9-13, 16-18, 21, and 26-27. Keep the university guide, completed forms, and credentials
outside Git.

## First login

Use the NTU campus network or the NTU VPN instructions in the supplied guide, then
log in with your approved lowercase NTU network account. First login creates the
home directory. Verify the SSH host fingerprint through university support and enter
passwords or MFA directly in the login interface.

Other members' access is not recorded; see [access status](../docs/TC2_ACCESS.md).

## Local network check

Use the actual head-node address from your guide:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\tc2_preflight.ps1 -HostName <head-node-address>
```

This only tests TCP port 22 and writes an ignored local JSON report. Exit 2 means
unreachable. Reachability does not prove account approval. After an interactive first
login and trusted host-key registration, optional existing-key authentication can be checked:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\tc2_preflight.ps1 -HostName <head-node-address> -Username <ntu-id> -CheckLogin
```

This uses BatchMode and strict host-key checking: it never obtains a password or
automatically accepts an unknown host. Exit 3 means the non-interactive login failed;
interactive/password authentication may still work. It only reads account/QoS info on
the head node. Use native `ssh`/`scp`/`sftp`; additional Windows transfer apps are optional.

## Inventory before choosing dependencies

On the cluster, upload the source/config/lock/scripts to your assigned directory. Do
not copy a local `.venv`, Git credentials, or another user's files. Transfer datasets and
checkpoints separately when needed, within the storage quota.
Plain filesystem management, Conda environment setup, and job submission follow the
guide's head-node workflow; GPU probing and application execution belong in compute jobs.

```bash
cd "$HOME/click2future"                     # use your actual uploaded repository path
mytcinfo
sacctmgr show user "$USER" withassoc format=user,qos
export TC2_QOS=normal                     # confirmed for Athul; verify for other accounts
sbatch --qos="$TC2_QOS" cluster/inventory.sbatch
squeue -u "$USER"
```

Use only a QoS assigned to you. All scripts deliberately omit a default QoS directive;
pass the verified assignment explicitly. The partition in the examples is `MGPU-TC2`;
confirm that it remains correct with your assignment. Inventory logs include the job/node,
driver/GPU memory, Linux/glibc, and available modules. The sample A40/driver screenshot
in the guide is dated April 2023 and is not current machine evidence.

## Prepare an isolated environment

The guide prefers Conda and forbids changing its shared base. Create your own small
bootstrap environment, using the available modules/channels and supported versions:

```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda create -n c2f-bootstrap -c conda-forge python=3.12 uv=0.13.0
conda activate c2f-bootstrap
uv --version
export C2F_UV="$(command -v uv)"
export C2F_PROJECT="$HOME/click2future"       # actual uploaded path
export C2F_BACKEND=cpu
sbatch --qos="$TC2_QOS" cluster/bootstrap.sbatch
```

The bootstrap job uses uv to create a project `.venv` from `.python-version` and
`uv.lock`, without modifying the shared base. Conda manages the bootstrap tools, and
uv manages the separately isolated project wheels. This is our proposed adaptation
of the guide's environment workflow, not a university-certified environment.
If the pinned uv package is unavailable from approved Conda channels, ask support or
install that missing tool into your own bootstrap environment as permitted by the guide.

Package/network availability, Linux wheel/glibc compatibility, and the driver must be
checked. Do not upgrade system CUDA, drivers, or glibc. The `cu128` lock was tested on Windows by its contributor; it must pass on the allocated Linux GPU too. Use an approved separate
cluster lock if the driver or OS cannot support it, and record that difference.
See [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

After inventory confirms that the CUDA 12.8 build is appropriate, switch the project backend:

```bash
export C2F_BACKEND=cu128
sbatch --qos="$TC2_QOS" cluster/bootstrap.sbatch
```

Wait for a successful installation/check report before submitting the profile job.
This download/build requires compute-node network access; if unavailable, arrange a
supported wheelhouse/installation route with support rather than repeatedly submit failing jobs.

## Verify the allocated GPU

```bash
sbatch --qos="$TC2_QOS" cluster/probe.sbatch
squeue -u "$USER"
sacct -u "$USER" --format=jobid,jobname,state,elapsed,alloctres
seff <completed-job-id>
```

The probe requires CUDA execution, exercises gradient/weight reload checks, and
profiles the untrained pilot on synthetic tensors. Reports go to `outputs/`; logs go
to the submission directory. Retrieve them with SFTP/SCP and record the actual allocation
in [COMPUTE.md](../docs/COMPUTE.md). Back up essential source, checkpoints, and run records;
the guide states that TC2 provides no data backup and allocations can expire.

The profiler supports `--require-slurm` and rejects execution without `SLURM_JOB_ID`.
This is an accidental-use guard, not evidence of account permissions. Timing a denoiser
does not measure a dataset pipeline, sampling quality, or complete training.

In standard Slurm, MaxWall limits each job and PENDING means waiting to start. Ordinary
Priority/Resources waits are not automatic failures. Check actual reason codes and
site-specific policies; the guide's pending-job wording is ambiguous. Training needs
full checkpoint/resume support before long jobs. [Slurm limits](https://slurm.schedmd.com/sacctmgr.html),
[job states](https://slurm.schedmd.com/job_state_codes.html).

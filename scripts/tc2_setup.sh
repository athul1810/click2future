#!/usr/bin/env bash
# macOS/Linux counterpart of scripts/tc2_setup.ps1: upload committed source and run
# cluster/run_setup.sh on the TC2 head node. Credentials are entered only in SSH prompts.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/tc2_setup.sh -u <ntu-username> [-s Inventory|Setup|Collect|All]
                            [-q <qos>] [-r <40-char-commit>] [-H <host>] [-n]
  -n  dry run: prepare the archive and remote script without connecting
EOF
    exit 2
}

username= host=10.96.189.12 stage=Inventory qos=normal revision= dry_run=0
while getopts 'u:H:s:q:r:n' opt; do
    case "$opt" in
        u) username=$OPTARG ;;
        H) host=$OPTARG ;;
        s) stage=$OPTARG ;;
        q) qos=$OPTARG ;;
        r) revision=$OPTARG ;;
        n) dry_run=1 ;;
        *) usage ;;
    esac
done
[[ "$username" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || usage
[[ "$host" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || usage
[[ "$qos" =~ ^[a-zA-Z0-9_-]+$ ]] || usage
case "$stage" in Inventory|Setup|Collect|All) ;; *) usage ;; esac
mode=$(printf '%s' "$stage" | tr '[:upper:]' '[:lower:]')

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$project_root"
for tool in git ssh scp shasum base64; do command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 2; }; done
if [[ -z "$revision" ]]; then revision=$(git rev-parse HEAD); else git cat-file -e "$revision^{commit}"; fi
[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo 'Cannot resolve source commit.' >&2; exit 2; }
for required in cluster/run_setup.sh cluster/setup.sbatch uv.lock; do
    git cat-file -e "$revision:$required" 2>/dev/null || { echo "Commit $revision does not contain $required." >&2; exit 2; }
done
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo 'Commit tracked changes before deployment; only the recorded commit is uploaded.' >&2
    exit 2
fi

local_dir="$project_root/outputs/tc2-setup/$revision"
mkdir -p "$local_dir"
archive_name="c2f-$revision.tar.gz"
archive_path="$local_dir/$archive_name"
target="$username@$host"
archive_hash=
if [[ "$stage" == Inventory || "$stage" == All ]]; then
    git -c core.autocrlf=false -c core.eol=lf archive --format=tar.gz --output="$archive_path" "$revision"
    archive_hash=$(shasum -a 256 "$archive_path" | cut -d ' ' -f 1)
fi

staging=
if [[ -n "$archive_hash" ]]; then
    staging=$(cat <<EOF
archive="\$HOME/$archive_name"
printf '%s  %s\n' '$archive_hash' "\$archive" | sha256sum -c -
mkdir -p "\$HOME/click2future/releases"
if [[ ! -e "\$release" ]]; then
    mkdir "\$release"
    tar -xzf "\$archive" -C "\$release" --no-same-owner
    mkdir -p "\$release/outputs"
    printf '%s\n' '$revision' > "\$release/outputs/source-revision.txt"
else
    printf '%s\n' 'Reusing this source release; existing source will not be overwritten.'
fi
EOF
)
fi
remote_script=$(cat <<EOF
set -euo pipefail
umask 077
release="\$HOME/click2future/releases/$revision"
$staging
if [[ ! -f "\$release/outputs/source-revision.txt" ]] || [[ "\$(cat "\$release/outputs/source-revision.txt")" != $revision ]]; then
    printf '%s\n' 'Source release is missing or incomplete. Run Inventory first.' >&2
    exit 2
fi
export C2F_PROJECT="\$release" C2F_QOS='$qos'
bash "\$release/cluster/run_setup.sh" $mode
EOF
)
printf '%s\n' "$remote_script" > "$local_dir/remote-$mode.sh"
echo "Target: $target | stage: $stage | QoS: $qos | source: $revision"
echo 'Only committed source is transferred. No Git credentials, local environment, datasets, or keys are uploaded.'
echo 'Enter NTU credentials only in SSH/SCP prompts. No transcript or password storage is used.'
if (( dry_run )); then
    echo "Dry run passed. Prepared commands are in $local_dir; no connection or job submission occurred."
    exit 0
fi

# One multiplexed connection, so the password is requested once for upload and run.
control_dir=$(mktemp -d "${TMPDIR:-/tmp}/c2f-ssh.XXXXXX")
ssh_opts=(-o StrictHostKeyChecking=ask -o ConnectTimeout=15 -o NumberOfPasswordPrompts=2
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3
    -o ControlMaster=auto -o "ControlPath=$control_dir/%C" -o ControlPersist=60)
cleanup() { ssh "${ssh_opts[@]}" -O exit "$target" 2>/dev/null || true; rm -rf -- "$control_dir"; }
trap cleanup EXIT

if [[ -n "$archive_hash" ]]; then
    scp "${ssh_opts[@]}" "$archive_path" "$target:$archive_name"
fi
encoded=$(printf '%s\n' "$remote_script" | base64 | tr -d '\n')
log_path="$local_dir/$mode.log"
bundle_b64="$local_dir/diagnostics-$mode.b64"
: > "$log_path"
: > "$bundle_b64"
set +e
ssh "${ssh_opts[@]}" "$target" "printf '%s' '$encoded' | base64 --decode | bash -l" | {
    reading=0 finished=0
    while IFS= read -r line; do
        if [[ "$line" == C2F_DIAGNOSTICS_BEGIN ]]; then reading=1
        elif [[ "$line" == C2F_DIAGNOSTICS_END ]]; then reading=0 finished=1
        elif (( reading )); then printf '%s' "$line" >> "$bundle_b64"
        else printf '%s\n' "$line" | tee -a "$log_path"
        fi
    done
    (( finished ))
}
statuses=("${PIPESTATUS[@]}")
set -e
remote_exit=${statuses[0]}
bundle_ok=$(( statuses[1] == 0 ))
if (( bundle_ok )); then
    base64 --decode < "$bundle_b64" > "$local_dir/diagnostics-$mode.tar.gz" 2>/dev/null \
        || base64 -D < "$bundle_b64" > "$local_dir/diagnostics-$mode.tar.gz"
    echo "Diagnostics saved: $local_dir/diagnostics-$mode.tar.gz"
fi
rm -f -- "$bundle_b64"
printf '{"checked_at_utc":"%s","stage":"%s","source_revision":"%s","remote_exit_code":%s,"diagnostics_downloaded":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$revision" "$remote_exit" \
    "$( (( bundle_ok )) && echo true || echo false)" > "$local_dir/$mode-result.json"
if (( remote_exit != 0 )); then echo "Remote stage failed (exit $remote_exit). Inspect the saved diagnostics." >&2; exit 1; fi
if (( ! bundle_ok )); then echo 'No completed diagnostic bundle was returned; setup remains unverified.' >&2; exit 1; fi

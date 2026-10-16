#!/usr/bin/env bash
# Head-node orchestration only: scheduler, filesystem, reports. No torch imports here.
set -euo pipefail
mode="${1:-inventory}"
: "${C2F_PROJECT:?Set the uploaded source directory}"
: "${C2F_QOS:?Set your assigned QoS explicitly}"
if [[ ! "$C2F_QOS" =~ ^[a-zA-Z0-9_-]+$ ]]; then exit 2; fi
cd -- "$C2F_PROJECT"
project_path=$(pwd -P)
home_path=$(cd -- "$HOME" && pwd -P)
case "$project_path/" in
    "$home_path/"*) ;;
    *) printf '%s\n' 'Source must be inside your own home directory.' >&2; exit 2 ;;
esac
mkdir -p outputs
umask 077
case "$mode" in
    inventory) job_script=cluster/inventory.sbatch ;;
    setup) job_script=cluster/setup.sbatch ;;
    collect) job_script= ;;
    *) printf '%s\n' 'Usage: run_setup.sh inventory|setup|collect' >&2; exit 2 ;;
esac
status=0
if [[ -n "$job_script" ]]; then
    printf 'account=%s source=%s qos=%s\n' "$USER" "$project_path" "$C2F_QOS"
    if command -v mytcinfo >/dev/null 2>&1; then mytcinfo; fi
    quota -s 2>/dev/null || true
    printf '%s\n' 'Submitting one GPU job; waiting for Slurm. Disconnecting does not cancel the submitted job.'
    set +e
    sbatch --parsable --wait --qos="$C2F_QOS" --export=ALL "$job_script" \
        | tee "outputs/tc2-$mode-submission.txt"
    status=${PIPESTATUS[0]}
    set -e
    job_id=$(head -n 1 "outputs/tc2-$mode-submission.txt" | cut -d ';' -f 1)
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES \
            | tee "outputs/tc2-$mode-accounting.txt" || true
        if command -v seff >/dev/null 2>&1; then seff "$job_id" || true; fi
        shopt -s nullglob
        for log in tc2-*"$job_id".out tc2-*"$job_id".err; do
            printf '\n--- %s ---\n' "$log"
            cat -- "$log"
        done
    fi
fi
# Return only setup diagnostics, not datasets, weights, credentials or unrelated files.
shopt -s nullglob
reports=(outputs/tc2-*.json outputs/tc2-*.csv outputs/tc2-*.txt outputs/tc2-*.xml tc2-*.out tc2-*.err)
if (( ${#reports[@]} )); then
    tar -czf outputs/tc2-diagnostics.tar.gz -- "${reports[@]}"
    printf '\n%s\n' 'C2F_DIAGNOSTICS_BEGIN'
    base64 outputs/tc2-diagnostics.tar.gz
    printf '%s\n' 'C2F_DIAGNOSTICS_END'
fi
exit "$status"

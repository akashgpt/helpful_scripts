#!/bin/bash
# auto_resubmit__RUN_VASP.sh
#
# Simple loop: submit SUBMIT_SCRIPT in PARENT_DIR, then sleep for SLEEP_SECONDS and
# check if it's still running. When it finishes, resubmit. Repeat until
# a stop file is created or no to_RUN markers remain.
#
# Usage:
#   # Run in background (or in a tmux/screen session):
#   nohup bash auto_resubmit__RUN_VASP.sh &
#
#   # To stop after current job finishes:
#   touch .stop_resubmit

# set -euo pipefail

# Everything is relative to the current directory
PARENT_DIR="$(pwd)"
shopt -s nullglob
submit_scripts=("${PARENT_DIR}"/MULTI_RUN_VASP*.sh)
shopt -u nullglob

if [[ "${#submit_scripts[@]}" -ne 1 ]]; then
    echo "Error: Expected exactly 1 submit script matching ${PARENT_DIR}/MULTI_RUN_VASP*.sh, found ${#submit_scripts[@]}."
    exit 1
fi

SUBMIT_SCRIPT="${submit_scripts[0]}"

# Stable tag = short dir name + checksum of full path.
# This lets a restarted controller keep tracking the same Slurm job.
PARENT_DIR_BASENAME="$(basename "$PARENT_DIR")"
PARENT_DIR_HASH="$(printf '%s' "$PARENT_DIR" | cksum | cut -d ' ' -f 1)"
JOB_TAG="${PARENT_DIR_BASENAME}_${PARENT_DIR_HASH}"

STOP_FILE="${PARENT_DIR}/.stop_resubmit"
LOG_FILE="${PARENT_DIR}/log.auto_resubmit"
LOCK_FILE="${PARENT_DIR}/.auto_resubmit.lock"
SLEEP_SECONDS=3600  # 60 minutes

# Redirect all stdout and stderr to log file (and still print to terminal)
exec > >(tee -a "$LOG_FILE") 2>&1

# Keep exactly one controller process per parent directory.
exec 9<>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date): Another auto_resubmit controller is already running for $PARENT_DIR. Exiting."
    exit 1
fi
: > "$LOCK_FILE"
printf "%s\n" "$$" 1>&9

echo "$(date): auto_resubmit started (PID $$)"
echo "  SUBMIT_SCRIPT: $SUBMIT_SCRIPT"
echo "  PARENT_DIR:    $PARENT_DIR"
echo "  JOB_TAG:       $JOB_TAG"
echo "  LOCK_FILE:     $LOCK_FILE"
echo "  Stop with:     touch $STOP_FILE"
echo ""

while true; do
    # ---- Check stop file ----
    if [[ -f "$STOP_FILE" ]]; then
        echo "$(date): Stop file found. Exiting resubmit loop."
        exit 0
    fi

    # ---- Check if our job is still running/pending ----
    existing=$(squeue -u "$USER" -n "$JOB_TAG" -h 2>/dev/null | wc -l)
    if [[ "$existing" -gt 0 ]]; then
        echo "$(date): Job '$JOB_TAG' still running/pending. Sleeping ${SLEEP_SECONDS}s..."
        sleep "$SLEEP_SECONDS"
        continue
    fi

    # ---- Count remaining to_RUN markers ----
    remaining=$(find "$PARENT_DIR" -name "to_RUN" -type f 2>/dev/null | head -500 | wc -l)
    if [[ "$remaining" -eq 0 ]]; then
        echo "$(date): No to_RUN markers remain. All done!"

        # check if "Elapsed" count in OUTCAR matches number of purely-numeric subfolders (1/, 2/, etc.)
        # Only look for OUTCARs inside those digit-named folders
        mapfile -t numeric_dirs < <(find "$PARENT_DIR" -type d -regex '.*/[0-9]+' 2>/dev/null | sort)
        total_runs=${#numeric_dirs[@]}
        completed_runs=0
        for ndir in "${numeric_dirs[@]}"; do
            if [[ -f "$ndir/OUTCAR" ]] && tail -20 "$ndir/OUTCAR" 2>/dev/null | grep -q "Elapsed"; then
                ((completed_runs++))
            fi
        done
        if [[ "$total_runs" -eq "$completed_runs" ]]; then
            echo "All runs completed successfully! Total runs: $total_runs."
        else
            echo "Warning: No to_RUN markers remain but not all runs have 'Elapsed' in log.run_sim. Total runs: $total_runs, Completed runs: $completed_runs."
        fi

        exit 0
    fi

    echo "$(date): $remaining to_RUN dirs remaining. Re-submitting..."

    # ---- Submit from PARENT_DIR with the tracking tag ----
    JOB_ID=$(sbatch --job-name="$JOB_TAG" "$SUBMIT_SCRIPT" 2>&1 | grep -oP '\d+')
    echo "$(date): Submitted job $JOB_ID (tag: $JOB_TAG)"

    # ---- Sleep before checking again ----
    sleep "$SLEEP_SECONDS"
done

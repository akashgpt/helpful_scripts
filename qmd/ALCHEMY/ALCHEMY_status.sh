#!/bin/bash
##############################################################################
# alchemy_status.sh — Status summary for ALCHEMY (TRAIN_MLMD_*) pipeline runs
#
# Usage:  ./alchemy_status.sh [-f] [sim_data_dir]
#   -f          Also write output to log.alchemy_status (plain text, no colors)
#   sim_data_dir  Directory to scan (default: directory containing this script)
#
# Parses log.TRAIN_MLMD_LEVEL_{1,2,2_v2,2_v3,3,4} files and prints a
# concise, structured status report for every iteration found.
#
# Author: Akash Gupta + Claude
##############################################################################


# ---------- parse flags ----------
WRITE_LOG=0
while getopts "f" opt; do
    case $opt in
        f) WRITE_LOG=1 ;;
        *) echo "Usage: $0 [-f] [sim_data_dir]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# ---------- locate sim_data_dir ----------
if [[ $# -ge 1 ]]; then
    SIM_DIR="$(cd "$1" && pwd)"
else
    SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# ---------- optionally tee output to log file (strip ANSI codes for the file) ----------
if [[ "$WRITE_LOG" -eq 1 ]]; then
    LOG_FILE="$SIM_DIR/log.alchemy_status"
    exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' > "$LOG_FILE")) 2>&1
fi

L1_LOG="$SIM_DIR/log.TRAIN_MLMD_LEVEL_1"
if [[ ! -f "$L1_LOG" ]]; then
    echo "ERROR: $L1_LOG not found. Provide the sim_data_dir as argument." >&2
    exit 1
fi

# ---------- helpers ----------
ts_to_epoch() {
    date -d "$1" +%s 2>/dev/null || echo ""
}

human_duration() {
    local secs=$1
    if [[ -z "$secs" || "$secs" -le 0 ]] 2>/dev/null; then echo "N/A"; return; fi
    local h=$((secs / 3600))
    local m=$(((secs % 3600) / 60))
    printf "%dh %02dm" "$h" "$m"
}

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

section() { echo -e "\n${BOLD}${CYAN}$1${NC}"; printf '%0.s─' $(seq 1 ${#1}); echo; }
ok()      { echo -e "  ${GREEN}✓${NC} $1"; }
warn()    { echo -e "  ${YELLOW}⏳${NC} $1"; }
err()     { echo -e "  ${RED}✗${NC} $1"; }
info()    { echo -e "  $1"; }

# Extract a timestamp from a line (handles "at <ts>", "on <ts>", etc.)
extract_ts() {
    echo "$1" | grep -oP '\w{3} \w{3} +\d+ [\d:]+ [AP]M \w+ \d{4}' | tail -1
}

# =====================================================================
#  LEVEL 1 — top-level orchestrator
# =====================================================================
section "ALCHEMY Pipeline Status — $SIM_DIR"
echo -e "  Report generated: $(date)"
echo ""

# Extract run header info — the L1 log has "### N iterations | PREFIX_i prefix | SYSTEM system ###"
HEADER_LINE=$(grep -P '\d+ iterations \|.*prefix \|.*system' "$L1_LOG" | tail -1)
RUN_PREFIX=$(echo "$HEADER_LINE" | grep -oP '\|\s*\K\S+(?=\s+prefix)' || true)
MOL_SYSTEM=$(echo "$HEADER_LINE" | grep -oP '\|\s*\K[A-Z][\w ]+(?=\s+system)' || true)
NUM_ITER=$(echo "$HEADER_LINE" | grep -oP '\d+(?=\s+iterations)' || true)

# Fallbacks
if [[ -z "$RUN_PREFIX" ]]; then
    # Try RUNID_PREFIX from the parameters block
    RUN_PREFIX=$(grep -A1 '^RUNID_PREFIX' "$L1_LOG" | tail -1 | xargs)
fi
if [[ -z "$MOL_SYSTEM" ]]; then
    MOL_SYSTEM=$(grep 'Molecular system being trained' "$L1_LOG" -A1 | tail -1 | xargs)
fi

SEQUENCE_PID=$(grep -oP '(?<=New Sequence of Iterations \[PID: )\d+' "$L1_LOG" | tail -1)
SEQUENCE_DATE=$(extract_ts "$(grep 'New Sequence of Iterations' "$L1_LOG" | tail -1)")

info "${BOLD}Run prefix:${NC}     ${RUN_PREFIX:-unknown}"
info "${BOLD}System:${NC}         ${MOL_SYSTEM:-unknown}"
info "${BOLD}Iterations:${NC}     ${NUM_ITER:-unknown}"
info "${BOLD}Sequence PID:${NC}   ${SEQUENCE_PID:-unknown}"
info "${BOLD}Sequence start:${NC} ${SEQUENCE_DATE:-unknown}"

# =====================================================================
#  Find all iterations
# =====================================================================
ITER_DIRS=()
if [[ -n "$RUN_PREFIX" ]]; then
    while IFS= read -r d; do
        [[ -d "$d" ]] && ITER_DIRS+=("$d")
    done < <(find "$SIM_DIR" -maxdepth 1 -type d -name "${RUN_PREFIX}*" | sort -V)
fi

if [[ ${#ITER_DIRS[@]} -eq 0 ]]; then
    echo ""
    err "No iteration directories found matching prefix '${RUN_PREFIX}*'"
    exit 1
fi

for ITER_DIR in "${ITER_DIRS[@]}"; do
    ITER_NAME=$(basename "$ITER_DIR")

    # =================================================================
    #  LEVEL 1 per-iteration info
    # =================================================================
    section "Iteration: $ITER_NAME"

    # L1 uses two forms: "Restarting from the last iteration <name>" and "Starting training iteration <name>"
    ITER_DONE_LINE=$(grep "Training iteration ${ITER_NAME} done" "$L1_LOG" 2>/dev/null || true)

    # Try multiple patterns for the start line
    ITER_START_LINE=$(grep -E "(Starting training iteration ${ITER_NAME}|Restarting from the last iteration ${ITER_NAME})" "$L1_LOG" 2>/dev/null | head -1 || true)
    ITER_START_TS=$(extract_ts "$ITER_START_LINE")
    ITER_DONE_TS=$(extract_ts "$ITER_DONE_LINE")

    if [[ -n "$ITER_DONE_LINE" ]]; then
        ok "Iteration complete"
        [[ -n "$ITER_START_TS" ]] && info "  Started:  $ITER_START_TS"
        [[ -n "$ITER_DONE_TS" ]] && info "  Finished: $ITER_DONE_TS"
        if [[ -n "$ITER_START_TS" && -n "$ITER_DONE_TS" ]]; then
            e1=$(ts_to_epoch "$ITER_START_TS"); e2=$(ts_to_epoch "$ITER_DONE_TS")
            if [[ -n "$e1" && -n "$e2" ]]; then
                info "  Duration: $(human_duration $((e2 - e1)))"
            fi
        fi
    else
        warn "Iteration IN PROGRESS"
        [[ -n "$ITER_START_TS" ]] && info "  Started: $ITER_START_TS"
    fi

    # =================================================================
    #  LEVEL 2 — DeePMD training + freeze
    # =================================================================
    L2_LOG="$ITER_DIR/log.TRAIN_MLMD_LEVEL_2"
    if [[ -f "$L2_LOG" ]]; then
        echo ""
        echo -e "  ${BOLD}[DeePMD Training & Freeze]${NC}"

        TRAIN_START=$(grep -oP 'DeepMD training \(\d+ w tag \S+\) started at \K.*' "$L2_LOG" | tail -1)
        TRAIN_DONE=$(grep -oP 'DeepMD training \(\d+ w tag \S+\) done at \K.*' "$L2_LOG" | tail -1 | sed 's/\.$//')
        TRAIN_JOB=$(grep -oP 'DeepMD training \(\K\d+' "$L2_LOG" | tail -1)
        FREEZE_DONE=$(grep -oP 'DeepMD freeze/compress done at \K.*' "$L2_LOG" | tail -1 | sed 's/\.$//')

        if [[ -n "$TRAIN_DONE" ]]; then
            ok "Training done (job ${TRAIN_JOB:-?})"
            [[ -n "$TRAIN_START" ]] && info "    Started:  $TRAIN_START"
            info "    Finished: $TRAIN_DONE"
            if [[ -n "$TRAIN_START" && -n "$TRAIN_DONE" ]]; then
                e1=$(ts_to_epoch "$TRAIN_START"); e2=$(ts_to_epoch "$TRAIN_DONE")
                if [[ -n "$e1" && -n "$e2" ]]; then
                    info "    Duration: $(human_duration $((e2 - e1)))"
                fi
            fi
        elif [[ -n "$TRAIN_START" ]]; then
            warn "Training in progress (job ${TRAIN_JOB:-?})"
            info "    Started: $TRAIN_START"
        else
            # Check if training was skipped (RESTART=-1 + done_deepmd_training for iteration 1)
            if grep -q 'Path 2.*MET' "$L2_LOG" 2>/dev/null; then
                info "  Training skipped (RESTART=-1, using existing model)"
            fi
        fi

        if [[ -n "$FREEZE_DONE" ]]; then
            ok "Freeze/compress done at $FREEZE_DONE"
        elif [[ -n "$TRAIN_DONE" ]]; then
            warn "Freeze/compress in progress"
        fi

        # L3 script launches
        L3_LAUNCHED=$(grep -c 'PID of this Level 3 script' "$L2_LOG" 2>/dev/null) || true
        L2_WAITING=$(grep 'Waiting for all TRAIN_MLMD_LEVEL_3.sh scripts to finish' "$L2_LOG" | tail -1 || true)
        L2_WAITING_TS=$(extract_ts "$L2_WAITING")
        L2_WAITING_COUNT=$(grep -c 'Waiting for all TRAIN_MLMD_LEVEL_3.sh scripts to finish' "$L2_LOG" 2>/dev/null) || true

        if [[ "$L3_LAUNCHED" -gt 0 ]]; then
            echo ""
            echo -e "  ${BOLD}[MD + Recal Phase (Level 3/4)]${NC}"
            info "  Level 3 scripts launched: $L3_LAUNCHED"
            if [[ -n "$L2_WAITING_TS" && -z "$ITER_DONE_LINE" ]]; then
                warn "Level 2 still polling ($L2_WAITING_COUNT checks so far, last: $L2_WAITING_TS)"
            fi
        fi
    fi

    # =================================================================
    #  Identify LAMMPS-MD log and VASP-recal log
    #  These may be in _v2, _v3, or the main L2 log depending on setup
    # =================================================================
    L2V2_LOG=""   # the one with LAMMPS-MD submitted/done
    L2V3_LOG=""   # the one with VASP-recal submitted/done

    for suffix in "" "_v2" "_v3"; do
        candidate="$ITER_DIR/log.TRAIN_MLMD_LEVEL_2${suffix}"
        [[ -f "$candidate" ]] || continue
        if [[ -z "$L2V2_LOG" ]] && grep -q 'LAMMPS-MD.*submitted' "$candidate" 2>/dev/null; then
            L2V2_LOG="$candidate"
        fi
        if [[ -z "$L2V3_LOG" ]] && grep -q 'VASP-recal.*submitted' "$candidate" 2>/dev/null; then
            L2V3_LOG="$candidate"
        fi
    done

    # =================================================================
    #  LAMMPS-MD jobs
    # =================================================================
    if [[ -n "$L2V2_LOG" ]]; then
        echo ""
        echo -e "  ${BOLD}[LAMMPS-MD Jobs]${NC} ($(basename "$L2V2_LOG"))"

        MD_SUBMITTED=$(grep -c 'LAMMPS-MD.*submitted' "$L2V2_LOG" 2>/dev/null) || true
        MD_DONE=$(grep -c 'LAMMPS-MD.*done' "$L2V2_LOG" 2>/dev/null) || true
        # Use the larger of submitted/done as the true total (restarts can cause mismatches)
        MD_TOTAL=$(( MD_SUBMITTED > MD_DONE ? MD_SUBMITTED : MD_DONE ))
        MD_FIRST_SUB=$(extract_ts "$(grep 'LAMMPS-MD.*submitted' "$L2V2_LOG" | head -1)")
        MD_LAST_DONE=$(extract_ts "$(grep 'LAMMPS-MD.*done' "$L2V2_LOG" | tail -1)")

        if [[ "$MD_DONE" -ge "$MD_SUBMITTED" && "$MD_SUBMITTED" -gt 0 ]]; then
            ok "All $MD_TOTAL MD jobs done"
        elif [[ "$MD_SUBMITTED" -gt 0 ]]; then
            warn "MD jobs: $MD_DONE/$MD_TOTAL done"
        fi
        [[ -n "$MD_FIRST_SUB" ]]  && info "    First submitted: $MD_FIRST_SUB"
        [[ -n "$MD_LAST_DONE" ]]  && info "    Last completed:  $MD_LAST_DONE"

        if [[ -n "$MD_FIRST_SUB" && -n "$MD_LAST_DONE" && "$MD_DONE" -ge "$MD_SUBMITTED" ]]; then
            e1=$(ts_to_epoch "$MD_FIRST_SUB"); e2=$(ts_to_epoch "$MD_LAST_DONE")
            if [[ -n "$e1" && -n "$e2" ]]; then
                info "    MD wall time:    $(human_duration $((e2 - e1)))"
            fi
        fi

        # Cross-cluster data returns (tracked in same log)
        DATA_BACK=$(grep -c 'Data back from the OTHER CLUSTER' "$L2V2_LOG" 2>/dev/null) || true
        DONE_WORKING=$(grep -c 'Done working on' "$L2V2_LOG" 2>/dev/null) || true
        if [[ "$DATA_BACK" -gt 0 ]]; then
            LAST_DATA_BACK=$(grep 'Data back from the OTHER CLUSTER' "$L2V2_LOG" | tail -1 | grep -oP '(?<=at \[).*(?=\])' || true)
            info "    Cross-cluster data returned: $DATA_BACK | Fully done: $DONE_WORKING"
            [[ -n "$LAST_DATA_BACK" ]] && info "    Last data return: $LAST_DATA_BACK"
        fi
    fi

    # =================================================================
    #  VASP recalculation jobs (cross-cluster, from a separate log)
    # =================================================================
    if [[ -n "$L2V3_LOG" && "$L2V3_LOG" != "$L2V2_LOG" ]]; then
        echo ""
        echo -e "  ${BOLD}[VASP Recalculation Jobs — cross-cluster]${NC} ($(basename "$L2V3_LOG"))"

        RECAL_SUBMITTED=$(grep -c 'VASP-recal.*submitted' "$L2V3_LOG" 2>/dev/null) || true
        RECAL_DONE=$(grep -c 'VASP-recal.*done' "$L2V3_LOG" 2>/dev/null) || true
        # Some configs get resubmitted, so count unique configs
        RECAL_UNIQUE=$(grep -oP 'VASP-recal.*(?:submitted|done)' "$L2V3_LOG" 2>/dev/null | sed 's/.*jobs for //;s/ submitted.*//;s/ done.*//' | sort -u | wc -l)
        RECAL_FIRST_SUB=$(extract_ts "$(grep 'VASP-recal.*submitted' "$L2V3_LOG" | head -1)")
        RECAL_LAST_DONE=$(extract_ts "$(grep 'VASP-recal.*done' "$L2V3_LOG" | tail -1)")

        RECAL_TOTAL=$(( RECAL_SUBMITTED > RECAL_DONE ? RECAL_SUBMITTED : RECAL_DONE ))
        RESUBS=$(( RECAL_TOTAL - RECAL_UNIQUE ))
        RESUB_NOTE=""
        [[ "$RESUBS" -gt 0 ]] && RESUB_NOTE=" (${RECAL_UNIQUE} configs, ${RESUBS} resubmissions)"

        if [[ "$RECAL_DONE" -ge "$RECAL_SUBMITTED" && "$RECAL_SUBMITTED" -gt 0 ]]; then
            ok "All ${RECAL_UNIQUE} recal jobs done${RESUB_NOTE}"
        elif [[ "$RECAL_SUBMITTED" -gt 0 ]]; then
            warn "Recal jobs: $RECAL_DONE/$RECAL_TOTAL done${RESUB_NOTE}"
        fi
        [[ -n "$RECAL_FIRST_SUB" ]]  && info "    First submitted: $RECAL_FIRST_SUB"
        [[ -n "$RECAL_LAST_DONE" ]]  && info "    Last completed:  $RECAL_LAST_DONE"

        if [[ -n "$RECAL_FIRST_SUB" && -n "$RECAL_LAST_DONE" && "$RECAL_DONE" -eq "$RECAL_SUBMITTED" ]]; then
            e1=$(ts_to_epoch "$RECAL_FIRST_SUB"); e2=$(ts_to_epoch "$RECAL_LAST_DONE")
            if [[ -n "$e1" && -n "$e2" ]]; then
                info "    Recal wall time: $(human_duration $((e2 - e1)))"
            fi
        fi
    fi

    # =================================================================
    #  LEVEL 3 / LEVEL 4 — per-zone/composition status
    #  Uses fast file-existence + tail-based checks to avoid slow greps
    # =================================================================
    MD_BASE="$ITER_DIR/md"
    if [[ -d "$MD_BASE" ]]; then
        echo ""
        echo -e "  ${BOLD}[Per-Zone/Composition Status]${NC}"

        TOTAL=0; DONE_L3=0; WAITING_RECAL=0; IN_MD=0; NOT_STARTED=0
        DONE_L4=0; TOTAL_L4=0
        ZONE_SUMMARY=""

        for ZONE_DIR in $(find "$MD_BASE" -maxdepth 1 -type d -name 'ZONE_*' | sort -V); do
            ZONE_NAME=$(basename "$ZONE_DIR")
            z_total=0; z_done=0; z_waiting=0; z_md=0

            for COMP_DIR in $(find "$ZONE_DIR" -maxdepth 1 -mindepth 1 -type d | sort); do
                COMP_NAME=$(basename "$COMP_DIR")
                L3="$COMP_DIR/log.TRAIN_MLMD_LEVEL_3"
                L4="$COMP_DIR/log.TRAIN_MLMD_LEVEL_4"

                z_total=$((z_total + 1))
                TOTAL=$((TOTAL + 1))

                if [[ -f "$L3" ]]; then
                    # Fast check: look at last 20 lines for "Done working on"
                    # (the log has trailing separator lines and blanks after the done message)
                    if tail -20 "$L3" | grep -q "Done working on" 2>/dev/null; then
                        z_done=$((z_done + 1))
                        DONE_L3=$((DONE_L3 + 1))
                    elif tail -20 "$L3" | grep -q 'CROSS_CLUSTER_MODE is ON' 2>/dev/null; then
                        # Only count as "waiting" if CROSS_CLUSTER is the last substantive action
                        z_waiting=$((z_waiting + 1))
                        WAITING_RECAL=$((WAITING_RECAL + 1))
                    else
                        z_md=$((z_md + 1))
                        IN_MD=$((IN_MD + 1))
                    fi
                else
                    NOT_STARTED=$((NOT_STARTED + 1))
                fi

                if [[ -f "$L4" ]]; then
                    TOTAL_L4=$((TOTAL_L4 + 1))
                    if tail -5 "$L4" | grep -q 'Recal phase.*successful' 2>/dev/null; then
                        DONE_L4=$((DONE_L4 + 1))
                    fi
                fi
            done

            # Per-zone one-liner
            if [[ $z_total -gt 0 ]]; then
                if [[ $z_done -eq $z_total ]]; then
                    ZONE_SUMMARY+="    ${GREEN}✓${NC} ${ZONE_NAME}: ${z_done}/${z_total} done\n"
                else
                    detail=""
                    [[ $z_done -gt 0 ]]    && detail+="${z_done} done, "
                    [[ $z_waiting -gt 0 ]] && detail+="${z_waiting} waiting recal, "
                    [[ $z_md -gt 0 ]]      && detail+="${z_md} in MD, "
                    detail="${detail%, }"
                    ZONE_SUMMARY+="    ${YELLOW}⏳${NC} ${ZONE_NAME}: ${z_done}/${z_total} (${detail})\n"
                fi
            fi
        done

        # Summary line
        if [[ $DONE_L3 -eq $TOTAL && $TOTAL -gt 0 ]]; then
            ok "All configurations complete: $DONE_L3/$TOTAL"
        elif [[ $TOTAL -gt 0 ]]; then
            warn "Configurations: $DONE_L3/$TOTAL done | $WAITING_RECAL waiting recal | $IN_MD in MD | $NOT_STARTED not started"
        fi

        if [[ $TOTAL_L4 -gt 0 ]]; then
            if [[ $DONE_L4 -eq $TOTAL_L4 ]]; then
                ok "VASP recal (L4): $DONE_L4/$TOTAL_L4 successful"
            else
                warn "VASP recal (L4): $DONE_L4/$TOTAL_L4 successful"
            fi
        fi

        # Print per-zone breakdown
        echo ""
        echo -e "$ZONE_SUMMARY" | sed '/^$/d'
    fi

    # =================================================================
    #  Error / Warning Scan
    # =================================================================
    echo ""
    echo -e "  ${BOLD}[Error / Warning Scan]${NC}"

    ERR_COUNT=0

    # Check for VASP recal reruns in L2 v3 log
    if [[ -n "$L2V3_LOG" && -f "$L2V3_LOG" ]]; then
        RERUNS=$(grep -c 'rerun of VASP-recal.*submitted' "$L2V3_LOG" 2>/dev/null) || true
        if [[ "$RERUNS" -gt 0 ]]; then
            RERUN_CONFIGS=$(grep -oP 'rerun of VASP-recal jobs for \K\S+' "$L2V3_LOG" 2>/dev/null | sort -u | tr '\n' ' ')
            warn "VASP recal reruns: $RERUNS (configs: ${RERUN_CONFIGS})"
            ERR_COUNT=$((ERR_COUNT + 1))
        fi
    fi

    # Check for LAMMPS errors in L3 logs (stuck simulations, missing Total wall time)
    if [[ -d "$MD_BASE" ]]; then
        LAMMPS_ERRS=0
        LAMMPS_ERR_LIST=""
        for ZONE_DIR in $(find "$MD_BASE" -maxdepth 1 -type d -name 'ZONE_*' 2>/dev/null); do
            ZONE_NAME=$(basename "$ZONE_DIR")
            for COMP_DIR in $(find "$ZONE_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null); do
                COMP_NAME=$(basename "$COMP_DIR")
                LOG_LAMMPS="$COMP_DIR/log.lammps"
                if [[ -f "$LOG_LAMMPS" ]] && ! grep -q 'Total wall time' "$LOG_LAMMPS" 2>/dev/null; then
                    # LAMMPS ran but didn't finish
                    LAMMPS_ERRS=$((LAMMPS_ERRS + 1))
                    LAMMPS_ERR_LIST+="$ZONE_NAME/$COMP_NAME "
                fi
            done
        done
        if [[ "$LAMMPS_ERRS" -gt 0 ]]; then
            warn "LAMMPS incomplete: $LAMMPS_ERRS configs missing 'Total wall time' in log.lammps"
            info "    $LAMMPS_ERR_LIST"
            ERR_COUNT=$((ERR_COUNT + 1))
        fi
    fi

    # Check for incomplete VASP OUTCARs (missing Elapsed time = crashed)
    # Only relevant when recal phase has started (VASP-recal jobs submitted)
    RECAL_PHASE_ACTIVE=0
    if [[ -n "$L2V3_LOG" && -f "$L2V3_LOG" ]]; then
        grep -q 'VASP-recal.*submitted' "$L2V3_LOG" 2>/dev/null && RECAL_PHASE_ACTIVE=1
    fi
    if [[ -d "$MD_BASE" && "$RECAL_PHASE_ACTIVE" -eq 1 ]]; then
        BAD_OUTCAR_LIST=$(find "$MD_BASE" -path "*/recal/*/OUTCAR" -print0 2>/dev/null | \
            xargs -0 grep -L 'Elapsed time' 2>/dev/null || true)
        BAD_OUTCARS=0
        [[ -n "$BAD_OUTCAR_LIST" ]] && BAD_OUTCARS=$(echo "$BAD_OUTCAR_LIST" | wc -l)
        if [[ "$BAD_OUTCARS" -gt 0 ]]; then
            err "Incomplete OUTCARs (no Elapsed time): $BAD_OUTCARS"
            ERR_COUNT=$((ERR_COUNT + 1))
        fi
    fi

    # Check for error_recal sentinel files (active VASP failures)
    if [[ -d "$MD_BASE" ]]; then
        ERR_RECAL_FILES=$(find "$MD_BASE" -name 'error_recal' 2>/dev/null)
        if [[ -n "$ERR_RECAL_FILES" ]]; then
            ERR_RECAL_COUNT=$(echo "$ERR_RECAL_FILES" | wc -l)
            warn "Active error_recal files: $ERR_RECAL_COUNT"
            echo "$ERR_RECAL_FILES" | while read -r ef; do
                info "    $ef"
            done
            ERR_COUNT=$((ERR_COUNT + 1))
        fi
    fi

    # Check L2 logs for any explicit error/failure messages
    # Exclude known false positives: "error_recal" (sentinel filename), "check log.lammps for any errors"
    for logf in "$L2_LOG" "$L2V2_LOG" "$L2V3_LOG"; do
        [[ -z "$logf" || ! -f "$logf" ]] && continue
        ERRS=$(grep -i -c -E 'ERROR|FATAL|FAILED|Traceback' "$logf" 2>/dev/null) || true
        # Subtract known false positives
        FP=$(grep -c -E 'error_recal|check.*for any errors|stale state files' "$logf" 2>/dev/null) || true
        ERRS=$((ERRS - FP))
        if [[ "$ERRS" -gt 0 ]]; then
            warn "$(basename "$logf"): $ERRS error/failure lines"
            grep -i -n -E 'ERROR|FATAL|FAILED|Traceback' "$logf" 2>/dev/null | \
                grep -v -E 'error_recal|check.*for any errors|stale state files' | \
                tail -3 | while read -r eline; do
                info "    $eline"
            done
            ERR_COUNT=$((ERR_COUNT + 1))
        fi
    done

    if [[ "$ERR_COUNT" -eq 0 ]]; then
        ok "No errors or warnings detected"
    fi

done

echo ""
section "End of Report"
echo ""

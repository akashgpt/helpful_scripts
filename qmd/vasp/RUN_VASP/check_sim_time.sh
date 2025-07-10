#!/usr/bin/env bash
# set -euo pipefail

# start from whatever root you like instead of "."
ROOT="."
counter_log_exists=0
counter_to_RUN_exists=0


find "$ROOT" -type f -name 'to_RUN__correction' -print0 |
while IFS= read -r -d '' runfile; do
    dir=$(dirname "$runfile")
    log_f="$dir/log.run_sim"
    OUTCAR_f="$dir/OUTCAR"

    counter_to_RUN_exists=$((counter_to_RUN_exists + 1))

    if [[ ! -f $log_f ]]; then
        echo "$dir → no log.run_sim found"
        continue
    fi
    if [[ ! -f $OUTCAR_f ]]; then
        echo "$dir → no OUTCAR found"
        continue
    fi

    # increment the counter for each found runfile
    counter_log_exists=$((counter_log_exists + 1))

    # grab the last line with "T=" and take its first field
    sim_time_logfile=$(grep 'T=' "$log_f" | tail -n1 | awk '{print $1}')
    sim_time_OUTCAR=$(grep "Iteration" "$OUTCAR_f" | tail -n 1 | awk '{print $3}')
    # if the last character is "(", remove it
    if [[ "$sim_time_OUTCAR" == *"("* ]]; then
        sim_time_OUTCAR=${sim_time_OUTCAR%"("*}
    fi

    # print directory and the time
    if [[ -z "$sim_time_logfile" ]]; then
        echo "$dir → no T= found in log.run_sim"
    else
        printf "%s : %s (log.run_sim)\n" "$dir" "$sim_time_logfile"
    fi

    if [[ -z "$sim_time_OUTCAR" ]]; then
        echo "$dir → no 'Iteration' entries found in OUTCAR"
    else
        printf "%s : %s (OUTCAR)\n" "$dir" "$sim_time_OUTCAR"
    fi
    echo ""
done

echo ""
echo ""

if (( counter_to_RUN_exists == 0 )); then
    echo "No 'to_RUN__correction' files found in $ROOT"
else
    echo "Total 'to_RUN__correction' files found: $counter_to_RUN_exists"
fi

if (( counter_log_exists == 0 )); then
    echo "No 'log.run_sim' files found in $ROOT"
else
    echo "Total 'log.run_sim' files found: $counter_log_exists"
fi
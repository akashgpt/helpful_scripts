#!/bin/bash
# =============================================================================
# ALCHEMY_recal_timing.sh — Analyze VASP recalculation and LAMMPS-MD timings.
#
# VASP:    "Elapsed time (sec)" from OUTCARs in recal/N directories
# LAMMPS:  "Total wall time" from log.lammps in md/ZONE_*/COMP/ directories
#
# Reports statistics by iteration, composition, and zone.
# Usage: ./ALCHEMY_recal_timing.sh [-f] [base_dir]
#   -f   Also write output to log.ALCHEMY_recal_timing (ANSI stripped)
# =============================================================================
# NOTE: Dedup by (iter, zone, comp, recal_n) to avoid double counting when
#       the same OUTCAR exists under both pre/recal/N/ and recal/N/.
BASE_DIR="${1:-.}"
FILE_LOG=false

# Parse flags
while [[ "$1" == -* ]]; do
    case "$1" in
        -f) FILE_LOG=true; shift ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done
[[ -n "$1" ]] && BASE_DIR="$1"

# Colors
BOLD='\033[1m'
NC='\033[0m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
DIM='\033[2m'

# Tee to file if -f
if $FILE_LOG; then
    LOG_FILE="$BASE_DIR/log.ALCHEMY_recal_timing"
    exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' > "$LOG_FILE"))
fi

# Temp files for data collection
TMPFILE_VASP=$(mktemp)
TMPFILE_MD=$(mktemp)
trap "rm -f $TMPFILE_VASP $TMPFILE_MD" EXIT

# =============================================================================
#  VASP recalculation timings
# =============================================================================
echo -e "${BOLD}Scanning VASP OUTCARs for elapsed times...${NC}"

# Single grep pass over all OUTCARs — no per-file subprocess spawning
# Each line: /path/to/recal/N/OUTCAR:   Elapsed time (sec):   29.550
grep -rH 'Elapsed time (sec):' --include=OUTCAR "$BASE_DIR" 2>/dev/null | \
    grep '/recal/[0-9]*/OUTCAR:' | \
    while IFS= read -r line; do
        filepath="${line%%:*}"
        elapsed="${line##*:}"
        elapsed="${elapsed// /}"  # strip whitespace

        # Parse path using bash parameter expansion (no subprocesses)
        # Path: .../v8_i1/md/ZONE_1/40H2_40NH3/pre/recal/34/OUTCAR
        recal_dir="${filepath%/OUTCAR}"        # .../recal/34
        recal_n="${recal_dir##*/}"              # 34
        recal_parent="${recal_dir%/*}"           # .../pre/recal or .../recal
        recal_parent="${recal_parent%/recal}"    # .../pre or .../comp
        recal_parent="${recal_parent%/pre}"      # .../comp (strip pre/ if present)
        comp="${recal_parent##*/}"               # 40H2_40NH3
        zone_dir="${recal_parent%/*}"            # .../ZONE_1
        zone="${zone_dir##*/}"                   # ZONE_1
        iter_path="${zone_dir%/md/*}"            # .../v8_i1
        iter="${iter_path##*/}"                  # v8_i1

        [[ "$iter" =~ ^v[0-9]+_i[0-9]+$ ]] || continue
        [[ "$zone" =~ ^ZONE_[0-9]+$ ]] || continue
        [[ -n "$comp" && -n "$recal_n" && "$elapsed" =~ ^[0-9.]+$ ]] || continue
        echo "$iter $zone $comp $recal_n $elapsed"
    done | awk '!seen[$1,$2,$3,$4]++' > "$TMPFILE_VASP"

VASP_TOTAL=$(wc -l < "$TMPFILE_VASP")
echo -e "${BOLD}Found $VASP_TOTAL VASP recal elapsed times.${NC}"

# =============================================================================
#  LAMMPS-MD timings
# =============================================================================
echo -e "${BOLD}Scanning LAMMPS log.lammps for wall times...${NC}"

# Single grep pass: "Total wall time: H:MM:SS" from log.lammps files
# Path: .../v8_i1/md/ZONE_1/40H2_40NH3/log.lammps
grep -rH 'Total wall time:' --include=log.lammps "$BASE_DIR" 2>/dev/null | \
    grep '/md/ZONE_[0-9]*/[^/]*/log\.lammps:' | \
    while IFS= read -r line; do
        filepath="${line%%:*}"
        # Extract H:MM:SS
        timestr="${line##*Total wall time: }"
        timestr="${timestr// /}"

        # Convert H:MM:SS to seconds
        IFS=: read -r h m s <<< "$timestr"
        elapsed=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))

        # Parse path: .../v8_i1/md/ZONE_1/40H2_40NH3/log.lammps
        comp_dir="${filepath%/log.lammps}"       # .../40H2_40NH3
        comp="${comp_dir##*/}"                    # 40H2_40NH3
        zone_dir="${comp_dir%/*}"                 # .../ZONE_1
        zone="${zone_dir##*/}"                    # ZONE_1
        iter_path="${zone_dir%/md/*}"             # .../v8_i1
        iter="${iter_path##*/}"                   # v8_i1

        [[ "$iter" =~ ^v[0-9]+_i[0-9]+$ ]] || continue
        [[ "$zone" =~ ^ZONE_[0-9]+$ ]] || continue
        [[ -n "$comp" && "$elapsed" =~ ^[0-9]+$ ]] || continue
        echo "$iter $zone $comp 0 $elapsed"
    done > "$TMPFILE_MD"

MD_TOTAL=$(wc -l < "$TMPFILE_MD")
echo -e "${BOLD}Found $MD_TOTAL LAMMPS-MD wall times.${NC}"
echo ""

if [[ "$VASP_TOTAL" -eq 0 && "$MD_TOTAL" -eq 0 ]]; then
    echo "No data found."
    exit 0
fi

# =============================================================================
#  Statistics function (shared awk for both VASP and MD)
# =============================================================================
run_stats() {
    local datafile="$1"
    local label="$2"
    [[ -s "$datafile" ]] || return

awk -v BOLD="$BOLD" -v NC="$NC" -v GREEN="$GREEN" -v YELLOW="$YELLOW" -v CYAN="$CYAN" -v DIM="$DIM" -v LABEL="$label" '
function fmt_time(s) {
    if (s < 60) return sprintf("%.1fs", s)
    if (s < 3600) return sprintf("%.1fm", s/60)
    return sprintf("%.1fh", s/3600)
}
function print_stats(label, n, sum, mn, mx,    avg) {
    if (n == 0) return
    avg = sum / n
    printf "  %-32s  n=%4d  mean=%8.1fs  min=%8.1fs  max=%8.1fs  total=%s\n", label, n, avg, mn, mx, fmt_time(sum)
}
{
    iter = $1; zone = $2; comp = $3; t = $5

    # Global
    g_n++; g_sum += t
    if (g_n == 1 || t < g_min) g_min = t
    if (g_n == 1 || t > g_max) g_max = t

    # Per iteration
    it_n[iter]++; it_sum[iter] += t
    if (!(iter in it_min) || t < it_min[iter]) it_min[iter] = t
    if (!(iter in it_max) || t > it_max[iter]) it_max[iter] = t

    # Per composition
    co_n[comp]++; co_sum[comp] += t
    if (!(comp in co_min) || t < co_min[comp]) co_min[comp] = t
    if (!(comp in co_max) || t > co_max[comp]) co_max[comp] = t

    # Per zone
    zo_n[zone]++; zo_sum[zone] += t
    if (!(zone in zo_min) || t < zo_min[zone]) zo_min[zone] = t
    if (!(zone in zo_max) || t > zo_max[zone]) zo_max[zone] = t

    # Per iter+comp
    ic = iter SUBSEP comp
    ic_n[ic]++; ic_sum[ic] += t
    if (!(ic in ic_min) || t < ic_min[ic]) ic_min[ic] = t
    if (!(ic in ic_max) || t > ic_max[ic]) ic_max[ic] = t

    # Per iter+zone
    iz = iter SUBSEP zone
    iz_n[iz]++; iz_sum[iz] += t
    if (!(iz in iz_min) || t < iz_min[iz]) iz_min[iz] = t
    if (!(iz in iz_max) || t > iz_max[iz]) iz_max[iz] = t

    # Track unique keys
    if (!seen_iter[iter]++) iters[++ni] = iter
    if (!seen_comp[comp]++) comps[++nc] = comp
    if (!seen_zone[zone]++) zones[++nz] = zone
}
END {
    # Sort iterations numerically: extract vN and iM, sort by (N, M)
    for (i = 1; i <= ni; i++) {
        match(iters[i], /v([0-9]+)_i([0-9]+)/, m)
        iter_v[i] = m[1] + 0; iter_i[i] = m[2] + 0
    }
    for (i = 1; i <= ni; i++) for (j = i+1; j <= ni; j++)
        if (iter_v[i] > iter_v[j] || (iter_v[i] == iter_v[j] && iter_i[i] > iter_i[j])) {
            tmp = iters[i]; iters[i] = iters[j]; iters[j] = tmp
            tmp = iter_v[i]; iter_v[i] = iter_v[j]; iter_v[j] = tmp
            tmp = iter_i[i]; iter_i[i] = iter_i[j]; iter_i[j] = tmp
        }
    # Sort compositions numerically by leading number (e.g. 0H2, 6H2, 17H2, ...)
    for (i = 1; i <= nc; i++) {
        match(comps[i], /^[0-9]+/); comp_num[i] = substr(comps[i], RSTART, RLENGTH) + 0
    }
    for (i = 1; i <= nc; i++) for (j = i+1; j <= nc; j++)
        if (comp_num[i] > comp_num[j]) {
            tmp = comps[i]; comps[i] = comps[j]; comps[j] = tmp
            tmp = comp_num[i]; comp_num[i] = comp_num[j]; comp_num[j] = tmp
        }
    # Sort zones numerically
    for (i = 1; i <= nz; i++) {
        match(zones[i], /[0-9]+/); zone_num[i] = substr(zones[i], RSTART, RLENGTH) + 0
    }
    for (i = 1; i <= nz; i++) for (j = i+1; j <= nz; j++)
        if (zone_num[i] > zone_num[j]) {
            tmp = zones[i]; zones[i] = zones[j]; zones[j] = tmp
            tmp = zone_num[i]; zone_num[i] = zone_num[j]; zone_num[j] = tmp
        }

    printf "%s══════════════════════════════════════════════════════════════════════════════%s\n", BOLD, NC
    printf "%s %s — GLOBAL SUMMARY%s\n", BOLD, LABEL, NC
    printf "%s══════════════════════════════════════════════════════════════════════════════%s\n", BOLD, NC
    print_stats("ALL", g_n, g_sum, g_min, g_max)
    printf "\n"

    # ── By Iteration ──
    printf "%s── By Iteration ──────────────────────────────────────────────────────────────%s\n", BOLD, NC
    for (i = 1; i <= ni; i++) {
        it = iters[i]
        print_stats(it, it_n[it], it_sum[it], it_min[it], it_max[it])
    }
    printf "\n"

    # ── By Composition ──
    printf "%s── By Composition ────────────────────────────────────────────────────────────%s\n", BOLD, NC
    for (i = 1; i <= nc; i++) {
        co = comps[i]
        print_stats(co, co_n[co], co_sum[co], co_min[co], co_max[co])
    }
    printf "\n"

    # ── By Zone ──
    printf "%s── By Zone ───────────────────────────────────────────────────────────────────%s\n", BOLD, NC
    for (i = 1; i <= nz; i++) {
        zo = zones[i]
        print_stats(zo, zo_n[zo], zo_sum[zo], zo_min[zo], zo_max[zo])
    }
    printf "\n"

    # ── Per Iteration × Composition ──
    printf "%s── Per Iteration × Composition ───────────────────────────────────────────────%s\n", BOLD, NC
    for (i = 1; i <= ni; i++) {
        it = iters[i]
        printf "  %s%s%s:\n", CYAN, it, NC
        for (j = 1; j <= nc; j++) {
            co = comps[j]
            ic = it SUBSEP co
            if (ic in ic_n)
                print_stats("    " co, ic_n[ic], ic_sum[ic], ic_min[ic], ic_max[ic])
        }
    }
    printf "\n"

    # ── Per Iteration × Zone ──
    printf "%s── Per Iteration × Zone ──────────────────────────────────────────────────────%s\n", BOLD, NC
    for (i = 1; i <= ni; i++) {
        it = iters[i]
        printf "  %s%s%s:\n", CYAN, it, NC
        for (j = 1; j <= nz; j++) {
            zo = zones[j]
            iz = it SUBSEP zo
            if (iz in iz_n)
                print_stats("    " zo, iz_n[iz], iz_sum[iz], iz_min[iz], iz_max[iz])
        }
    }
}
' "$datafile"
}

# Run stats for both VASP and LAMMPS-MD
run_stats "$TMPFILE_VASP" "VASP Recalculation"
echo ""
echo ""
run_stats "$TMPFILE_MD" "LAMMPS-MD"

# =============================================================================
#  [Error / Warning Scan] — LAMMPS restart report
# =============================================================================
# Restarts are indicated by log.lammps.retry_{N}_backup files.
# When a LAMMPS-MD run fails (e.g. lost atoms), the script halves the timestep
# and retries, backing up the failed log as log.lammps.retry_{N}_backup.
echo ""
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD} [Error / Warning Scan] — LAMMPS-MD Restarts${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════════════════════${NC}"

# Collect all retry backup files: path pattern .../v8_i1/md/ZONE_1/COMP/log.lammps.retry_N_backup
TMPFILE_RETRIES=$(mktemp)
trap "rm -f $TMPFILE_VASP $TMPFILE_MD $TMPFILE_RETRIES" EXIT

find "$BASE_DIR" -name 'log.lammps.retry_*_backup' -path '*/md/ZONE_*/*/log.lammps.retry_*_backup' 2>/dev/null | \
    while IFS= read -r filepath; do
        # Extract retry number from filename
        fname="${filepath##*/}"                         # log.lammps.retry_1_backup
        retry_n="${fname#log.lammps.retry_}"             # 1_backup
        retry_n="${retry_n%_backup}"                      # 1

        # Parse path: .../v8_i1/md/ZONE_1/40H2_40NH3/log.lammps.retry_1_backup
        comp_dir="${filepath%/*}"                        # .../40H2_40NH3
        comp="${comp_dir##*/}"                           # 40H2_40NH3
        zone_dir="${comp_dir%/*}"                        # .../ZONE_1
        zone="${zone_dir##*/}"                           # ZONE_1
        iter_path="${zone_dir%/md/*}"                    # .../v8_i1
        iter="${iter_path##*/}"                          # v8_i1

        [[ "$iter" =~ ^v[0-9]+_i[0-9]+$ ]] || continue
        [[ "$zone" =~ ^ZONE_[0-9]+$ ]] || continue
        [[ -n "$comp" && "$retry_n" =~ ^[0-9]+$ ]] || continue
        echo "$iter $zone $comp $retry_n"
    done > "$TMPFILE_RETRIES"

RETRY_TOTAL=$(wc -l < "$TMPFILE_RETRIES")

if [[ "$RETRY_TOTAL" -eq 0 ]]; then
    echo -e "  ${GREEN}No LAMMPS restarts detected.${NC}"
else
    # Use awk to summarize:
    #   - total sims restarted (unique iter+zone+comp)
    #   - total retry count
    #   - max retries for any single sim
    #   - breakdown by iteration, then by zone+comp with retry count
    awk -v BOLD="$BOLD" -v NC="$NC" -v GREEN="$GREEN" -v YELLOW="$YELLOW" -v CYAN="$CYAN" -v DIM="$DIM" '
    {
        iter = $1; zone = $2; comp = $3; retry_n = $4 + 0
        key = iter SUBSEP zone SUBSEP comp

        # Track max retry number per sim (retry_1 = 1 restart, retry_2 = 2 restarts, etc.)
        if (!(key in max_retry) || retry_n > max_retry[key]) max_retry[key] = retry_n
        total_retries++

        # Track unique iterations
        if (!seen_iter[iter]++) iters[++ni] = iter

        # Track per-iteration sims
        ik = iter SUBSEP zone SUBSEP comp
        if (!seen_ik[ik]++) {
            iter_sims[iter]++
        }
    }
    END {
        # Count unique sims and find overall max retries
        overall_max = 0
        for (k in max_retry) {
            unique_sims++
            if (max_retry[k] > overall_max) overall_max = max_retry[k]
        }

        # Sort iterations
        for (i = 1; i <= ni; i++) {
            match(iters[i], /v([0-9]+)_i([0-9]+)/, m)
            iter_v[i] = m[1] + 0; iter_i[i] = m[2] + 0
        }
        for (i = 1; i <= ni; i++) for (j = i+1; j <= ni; j++)
            if (iter_v[i] > iter_v[j] || (iter_v[i] == iter_v[j] && iter_i[i] > iter_i[j])) {
                tmp = iters[i]; iters[i] = iters[j]; iters[j] = tmp
                tmp = iter_v[i]; iter_v[i] = iter_v[j]; iter_v[j] = tmp
                tmp = iter_i[i]; iter_i[i] = iter_i[j]; iter_i[j] = tmp
            }

        printf "\n"
        printf "  %sTotal simulations restarted:%s  %d\n", BOLD, NC, unique_sims
        printf "  %sTotal retry backup files:%s     %d\n", BOLD, NC, total_retries
        printf "  %sMax retries (any single sim):%s %d\n", BOLD, NC, overall_max
        printf "\n"

        # Breakdown by iteration
        printf "%s── Restarts by Iteration ─────────────────────────────────────────────────────%s\n", BOLD, NC
        for (i = 1; i <= ni; i++) {
            it = iters[i]
            printf "  %s%s%s:  %d sim(s) restarted\n", CYAN, it, NC, iter_sims[it]
        }
        printf "\n"

        # Detailed list: group by iteration, show zone/comp and retry count
        printf "%s── Restart Details (per simulation) ──────────────────────────────────────────%s\n", BOLD, NC

        for (i = 1; i <= ni; i++) {
            it = iters[i]
            printf "  %s%s%s:\n", CYAN, it, NC

            # Collect zone+comp keys for this iteration, sort, and print
            delete items; nk = 0
            for (k in max_retry) {
                split(k, parts, SUBSEP)
                if (parts[1] == it) {
                    nk++
                    items[nk] = parts[2] "/" parts[3] SUBSEP max_retry[k]
                }
            }
            # Sort items alphabetically
            for (a = 1; a <= nk; a++) for (b = a+1; b <= nk; b++)
                if (items[a] > items[b]) { tmp = items[a]; items[a] = items[b]; items[b] = tmp }

            for (a = 1; a <= nk; a++) {
                split(items[a], p, SUBSEP)
                retries = p[2] + 0
                color = (retries >= 2) ? YELLOW : DIM
                printf "    %s%-40s  retries: %d%s\n", color, p[1], retries, NC
            }
        }
    }
    ' "$TMPFILE_RETRIES"
fi

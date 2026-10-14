#!/bin/bash
# Non-HPC counterpart of benchmark/slurm/matrix.sh --suite=comparison, for a single workstation or
# laptop (no SLURM, no NUMA, no root). Runs benchmark/comparison/scripts/run_all.jl once per
# (BLAS backend, thread count), one run at a time, each pinned with `numactl`/`taskset` to its own
# physical cores (never an SMT sibling, same rule as _setup.jl). Results go where the SLURM runs
# put them, benchmark/comparison/results/runs/, plus a log per run under benchmark/results/local/.
#
# usage:
#   benchmark/local/run_comparison.sh [--threads=1,2] [--backends=openblas] [--prepare]
#                                     [--small] [--no-pin] [run_all.jl flags...]
#
#   --threads=LIST    thread counts, comma separated (default: 1..N-physical-cores, powers of two)
#   --backends=LIST   openblas and/or mkl (default: openblas)
#   --prepare         instantiate the Julia environment, then exit
#   --small           RISTRETTO_BENCH_SMALL=1 smoke sizes (minutes instead of hours)
#   --smt             allow SMT siblings: thread counts up to the logical CPU count (default list then
#                     is just that count), tagged placement "shared" so they never mix with physical-core runs
#   --cooldown=SEC    pause between consecutive runs (never before the first or after the last; default 0)
#   --cooldown-temp=C after the --cooldown pause, keep waiting until the hottest sensor is at most C degrees
#                     (at most --cooldown-max seconds, default 900), so every run starts equally cool
#   --bart-modes=LIST estimate and/or measure (default both): BART planning its FFTs with FFTW_ESTIMATE
#                     (its default) and with FFTW_MEASURE, as separate rows. Each mode repeats the other
#                     toolkits' rows too, so `estimate` alone halves the time.
#   --session=DIR     keep the logs and the unit list (units.tsv: status and time per unit) in DIR
#   --resume[=DIR]    continue a session (the newest one by default): done units are skipped, failed and
#                     interrupted ones run again. Pass the same options as the first time.
#   --plan-only       write/print the unit list and exit (status 10 when there is nothing left to run)
#   --no-pin          do not wrap the run in numactl/taskset
# Everything else (--sections=, --cases=, --frameworks=, --remeasure, --data=...) is passed to run_all.jl.
#
# Optional machine paths (BART builds, SigPy interpreter, data cache) are read from
# benchmark/slurm/site.env when that file exists, exactly as on the cluster; BART, SigPy and MRpro
# rows are skipped with a warning when they are not configured.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [ -f benchmark/slurm/site.env ]; then
    set -a
    # shellcheck source=/dev/null
    source benchmark/slurm/site.env
    set +a
fi
JULIA_BIN="${JULIA_BIN:-$(command -v julia || true)}"
[ -x "${JULIA_BIN:-}" ] || JULIA_BIN="$HOME/.juliaup/bin/julia"   # juliaup shim: not on PATH under sudo/runuser
[ -x "${JULIA_BIN:-}" ] || { echo "### no julia on PATH: set JULIA_BIN" >&2; exit 1; }
PROJECT_DIR=benchmark/comparison
SCRIPT_PATH=benchmark/comparison/scripts/run_all.jl

# One CPU per physical core (the lowest-numbered sibling).
mapfile -t PHYS_CPUS < <(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -k1,1n | awk -F, '!seen[$2]++ {print $1}' | sort -n)
N_PHYS=${#PHYS_CPUS[@]}
mapfile -t ALL_CPUS < <(lscpu -p=CPU | grep -v '^#' | sort -n)

THREADS=()
BACKENDS=(openblas)
PREPARE=0
PIN=1
SMT=0
COOLDOWN=0
SESSION=""
RESUME=0
PLAN_ONLY=0
COOL_TEMP=
COOL_MAX=900
BART_MODES=(estimate measure)
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --threads=*) IFS=, read -ra THREADS <<<"${a#*=}" ;;
        --backends=*) IFS=, read -ra BACKENDS <<<"${a#*=}" ;;
        --prepare) PREPARE=1 ;;
        --small) export RISTRETTO_BENCH_SMALL=1 ;;
        --cooldown=*) COOLDOWN="${a#*=}" ;;
        --cooldown-temp=*) COOL_TEMP="${a#*=}" ;;
        --cooldown-max=*) COOL_MAX="${a#*=}" ;;
        --bart-modes=*) IFS=, read -ra BART_MODES <<<"${a#*=}" ;;
        --session=*) SESSION="${a#*=}" ;;
        --resume) RESUME=1 ;;
        --resume=*) RESUME=1; SESSION="${a#*=}" ;;
        --plan-only) PLAN_ONLY=1 ;;
        --smt) SMT=1 ;;
        --no-pin) PIN=0 ;;
        *) PASS_ARGS+=("$a") ;;
    esac
done
if [ "$SMT" = 1 ]; then
    # lowest-numbered siblings first, then their siblings
    CPU_ORDER=("${PHYS_CPUS[@]}")
    for c in "${ALL_CPUS[@]}"; do [[ " ${PHYS_CPUS[*]} " == *" $c "* ]] || CPU_ORDER+=("$c"); done
    N_MAX=${#ALL_CPUS[@]}
    export RISTRETTO_BENCH_ALLOW_SMT=1 RISTRETTO_BENCH_PLACEMENT=shared
else
    CPU_ORDER=("${PHYS_CPUS[@]}")
    N_MAX=$N_PHYS
fi
if [ "${#THREADS[@]}" = 0 ]; then
    if [ "$SMT" = 1 ]; then
        THREADS=("$N_MAX")   # the physical-core counts are what a run without --smt measures
    else
        for ((t = 1; t <= N_MAX; t *= 2)); do THREADS+=("$t"); done
        [ "${THREADS[-1]}" != "$N_MAX" ] && THREADS+=("$N_MAX")
    fi
fi

if [ "$PREPARE" = 1 ]; then
    exec "$JULIA_BIN" --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
fi

for t in "${THREADS[@]}"; do
    if [ "$t" -gt "$N_MAX" ]; then
        echo "### $t threads requested but only $N_MAX usable CPUs (use --smt for hyperthreads)" >&2
        exit 2
    fi
done

# Same process-wide settings as slurm/common.sh.
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export TMPDIR="${TMPDIR_BENCH:-/dev/shm}"
export RISTRETTO_BENCH_PLACEMENT="${RISTRETTO_BENCH_PLACEMENT:-isolated}"

# A session directory holds the logs and units.tsv, the list of units with their status and time.
# --session=DIR works in DIR (new or existing); --resume[=DIR] continues one, the newest by default.
if [ "$RESUME" = 1 ] && [ -z "$SESSION" ]; then
    SESSION="$(ls -td "$REPO_ROOT"/benchmark/results/local/*/ 2>/dev/null | while read -r d; do [ -s "${d}units.tsv" ] && { echo "${d%/}"; break; }; done)"
    [ -n "$SESSION" ] || { echo "### nothing to resume: no session with a unit list" >&2; exit 1; }
fi
if [ -z "$SESSION" ]; then SESSION="$REPO_ROOT/benchmark/results/local/$(date +%Y%m%d_%H%M%S)"; fi
LOG_DIR="$SESSION"
UNITS_FILE="$LOG_DIR/units.tsv"
[ "$RESUME" = 1 ] && [ ! -s "$UNITS_FILE" ] && { echo "### $LOG_DIR has no units.tsv to resume" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$REPO_ROOT/benchmark/comparison/results/runs"
{
    echo "### $(date -Is) on $(hostname)"
    echo "### threads: ${THREADS[*]}  backends: ${BACKENDS[*]}  pass-through: ${PASS_ARGS[*]:-<none>}"
    echo "### physical cpus: ${PHYS_CPUS[*]}"
    lscpu | grep -E 'Model name|MHz'
    for f in /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/intel_pstate/no_turbo /sys/devices/system/cpu/smt/control; do
        [ -r "$f" ] && echo "### $f: $(cat "$f")"
    done
} | tee -a "$LOG_DIR/environment.txt"

# Sections run one by one (each is its own subprocess anyway), so a cooldown can sit between them.
SECTIONS=(base noncart cgsense sparsity dynamic kspace accuracy_race)
REST=()
for a in "${PASS_ARGS[@]}"; do
    case "$a" in
        --sections=*) IFS=, read -ra SECTIONS <<<"${a#*=}" ;;
        *) REST+=("$a") ;;
    esac
done

# BART modes: "estimate" is BART's default FFTW_ESTIMATE planning; "measure" sets
# RISTRETTO_BENCH_BART_MEASURE=1 (FFTW_MEASURE, wisdom emptied before every run) and labels its rows
# "BART (<backend>, MEASURE)". Without a BART build the modes would measure the same thing: one pass.
bart_bin() { [ "$1" = mkl ] && echo "${RISTRETTO_BENCH_BART_MKL:-}" || echo "${RISTRETTO_BENCH_BART_OPENBLAS:-}"; }

UNITS=()   # threads:backend:mode:section
for t in "${THREADS[@]}"; do
    for b in "${BACKENDS[@]}"; do
        modes=("${BART_MODES[@]}")
        [ -f "$(bart_bin "$b")" ] || modes=(estimate)
        for m in "${modes[@]}"; do
            for sec in "${SECTIONS[@]}"; do
                # MEASURE changes only BART, so that pass times BART alone; base and kspace have no timed BART rows
                [ "$m" = measure ] && [[ "$sec" == base || "$sec" == kspace ]] && continue
                UNITS+=("$t:$b:$m:$sec")
            done
        done
    done
done
echo "### ${#UNITS[@]} runs, cooldown ${COOLDOWN} s between them; bart modes: ${BART_MODES[*]}" | tee -a "$LOG_DIR/environment.txt"

# ---- unit list: status per unit, kept in units.tsv (unit, status, seconds, finished) ------------------
touch "$UNITS_FILE"
set_unit() {   # unit status seconds
    local tmp; tmp="$(mktemp "$LOG_DIR/.units.XXXXXX")"
    awk -F'\t' -v OFS='\t' -v u="$1" -v s="$2" -v d="$3" -v t="$(date +%H:%M:%S)" '$1 == u {$2 = s; $3 = d; $4 = t} {print}' "$UNITS_FILE" >"$tmp" && mv "$tmp" "$UNITS_FILE"
}
for u in "${UNITS[@]}"; do
    if ! awk -F'\t' -v u="$u" '$1 == u {f = 1} END {exit !f}' "$UNITS_FILE"; then
        printf '%s\tpending\t0\t\n' "$u" >>"$UNITS_FILE"
    elif [ "$RESUME" = 1 ]; then
        # interrupted units start over; done ones stay done, failed ones are retried
        [ "$(awk -F'\t' -v u="$u" '$1 == u {print $2}' "$UNITS_FILE")" = done ] || set_unit "$u" pending 0
    else
        set_unit "$u" pending 0
    fi
done
TODO=0
for u in "${UNITS[@]}"; do
    [ "$(awk -F'\t' -v u="$u" '$1 == u {print $2}' "$UNITS_FILE")" = done ] || TODO=$((TODO + 1))
done
echo "### session: $LOG_DIR ($TODO of ${#UNITS[@]} units to run)"
"$HERE/units_report.sh" "$LOG_DIR"
if [ "$PLAN_ONLY" = 1 ]; then [ "$TODO" -gt 0 ] && exit 0 || exit 10; fi

# Hottest hwmon sensor, in whole degrees C.
max_temp() { local f m=0 t; for f in /sys/class/hwmon/hwmon*/temp*_input; do
    t="$(cat "$f" 2>/dev/null)"   # a sensor can be unreadable or empty
    [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -gt "$m" ] && m=$t; done; echo "$((m / 1000))"; }

FAILED=0
first=1
for u in "${UNITS[@]}"; do
    IFS=: read -r t b m sec <<<"$u"
    if [ "$(awk -F'\t' -v u="$u" '$1 == u {print $2}' "$UNITS_FILE")" = done ]; then
        echo "### skipping $u: done already"
        continue
    fi
    if [ "$first" = 0 ]; then
        echo "### $(date +%H:%M:%S) temperature at the end of the last run: $(max_temp) C"
        [ "$COOLDOWN" -gt 0 ] && { echo "### cooldown ${COOLDOWN} s"; sleep "$COOLDOWN"; }
        if [ -n "$COOL_TEMP" ]; then
            waited=0
            while [ "$(max_temp)" -gt "$COOL_TEMP" ] && [ "$waited" -lt "$COOL_MAX" ]; do sleep 5; waited=$((waited + 5)); done
            echo "### waited ${waited} s more for <= ${COOL_TEMP} C (max ${COOL_MAX} s)"
        fi
        echo "### $(date +%H:%M:%S) temperature at the start of the next run: $(max_temp) C"
    else
        echo "### temperature at the start of the first run: $(max_temp) C"
    fi
    first=0
    args=(--threads="$t" --sections="$sec")
    [ "$b" = mkl ] && args+=(--use-mkl)
    cpus="$(IFS=,; echo "${CPU_ORDER[*]:0:t}")"
    pin=()
    if [ "$PIN" = 1 ]; then
        if command -v numactl >/dev/null; then pin=(numactl --physcpubind="$cpus" --membind=0)
        elif command -v taskset >/dev/null; then pin=(taskset -c "$cpus"); fi
    fi
    measure=0; [ "$m" = measure ] && { measure=1; args+=(--frameworks=bart); }
    log="$LOG_DIR/${b}_${t}t_${m}_${sec}.log"
    echo "### $(date +%H:%M:%S) $b, $t thread(s), bart $m, section $sec, cpus $cpus -> $log"
    set_unit "$u" running 0
    unit_start=$SECONDS
    "${pin[@]}" env JULIA_NUM_THREADS="$t" OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" MKL_NUM_THREADS="$t" \
        RISTRETTO_BENCH_BART_MEASURE="$measure" \
        "$JULIA_BIN" --project="$PROJECT_DIR" -t "$t" "$SCRIPT_PATH" "${args[@]}" "${REST[@]}" >"$log" 2>&1
    rc=$?
    # run_all.jl only warns when a section dies, so the exit status is not enough: a unit failed when it
    # exited with an error or reported a failed section. A clean run that recorded nothing (the case
    # filter matched no case of this section) counts as done, with a note.
    if [ "$rc" = 0 ] && ! grep -q "section $sec failed" "$log"; then
        grep -q "flushed run" "$log" || echo "### note: $u recorded no rows (no case of this section matched?)"
        set_unit "$u" done $((SECONDS - unit_start))
    else
        echo "### $b/$t/$m/$sec failed (see $log)" >&2
        set_unit "$u" failed $((SECONDS - unit_start)); FAILED=1
    fi
    "$HERE/units_report.sh" "$LOG_DIR" brief
done
"$HERE/units_report.sh" "$LOG_DIR"
echo "### done $(date +%H:%M:%S); results: benchmark/comparison/results/runs/, logs: $LOG_DIR"
exit $FAILED

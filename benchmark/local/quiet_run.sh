#!/bin/bash
# Quiet-machine wrapper for benchmark/local/run_comparison.sh, tailored to this laptop
# (CachyOS, KDE Plasma via plasmalogin, i5-5300U: 2 cores / 4 threads, 7 GB RAM, intel_pstate passive).
#
# What it does, as root, in a detached transient unit that outlives the desktop:
#   1. stops KDE (plasmalogin) and every process of your user, then the non-essential services
#      (auto-cpufreq, ananicy-cpp, bluetooth, cups, avahi, udisks2, upower, rtkit, timesyncd, ...);
#      sshd, NetworkManager/iwd, systemd-oomd (OOM protection with 7 GB) and journald stay up
#      unless --no-network is given
#   2. fixes the CPU: governor=performance, min freq = max freq (no frequency hunting), turbo off
#      (thermal-limited laptop: turbo timings drift with the heat), NMI watchdog off,
#      page cache dropped
#      SMT is switched per thread count (see below): off for counts up to the physical cores, on above.
#   3. switches to a status screen on virtual console 9 (quiet_status.sh: frequency, temperature,
#      throttle count, load, the benchmark's own log, refreshed every 2 s)
#   4. runs run_comparison.sh as the normal user, niced, with the arguments given here
#   5. whatever happens (finish, abort, failure), restores every setting and restarts services and KDE.
#
# Abort at any time: press q (then y) on the status screen, or Ctrl+C there. That kills the benchmark
# and brings the desktop back. From elsewhere (ssh, another VT): sudo touch /run/ristretto-quiet/abort
# or sudo systemctl stop ristretto-quiet. If KDE still does not come back: sudo systemctl start plasmalogin
#
# SMT: thread counts up to the number of physical cores (1 and 2 here) run with SMT off, counts above it
# (4 here) with SMT on, pinned to the siblings; the default counts are 1,2,4. --smt-mode=on keeps SMT on
# for every count, --smt-mode=off drops the counts above the physical cores. --threads=LIST overrides the counts.
#
# Resume: every run is a session (benchmark/results/local/session_<stamp>/) with a unit list, units.tsv:
# one line per (threads, backend, BART mode, section) with its status and time. `--resume` continues the
# newest session with the arguments it was started with: done units are skipped, failed and interrupted ones
# run again. `--resume=DIR` names a session. Print the list any time:
#   benchmark/local/units_report.sh benchmark/results/local/session_<stamp>
#
# usage:   benchmark/local/quiet_run.sh [--resume[=DIR]] [--no-network] [--smt-mode=auto|on|off] [--threads=LIST] [--vt=N] [--cooldown=SEC] [--cooldown-temp=C] [run_comparison.sh flags...]
#   e.g.   benchmark/local/quiet_run.sh --threads=1,2 --backends=openblas
#          benchmark/local/quiet_run.sh --threads=4    (SMT on, 4 threads on 2 cores)
# Needs sudo (asked once, up front). Everything on the desktop is closed WITHOUT saving: save your
# work first. It detaches immediately; the terminal you run it from dies with KDE.
set -u

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SELF")"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DM=plasmalogin.service
# Stopped for the run, started again afterwards if they were active before.
NOISY=(auto-cpufreq ananicy-cpp bluetooth cups cups-browsed avahi-daemon avahi-daemon.socket
       cups.socket cups.path udisks2 upower rtkit-daemon systemd-timesyncd)
NETWORK=(NetworkManager iwd systemd-resolved)
RUN=/run/ristretto-quiet

if [ "${1:-}" != "--inner" ]; then
    # ---- outer: runs as the user, in the desktop ------------------------------------------------
    [ "$(id -u)" = 0 ] && { echo "run as your normal user, it uses sudo itself" >&2; exit 1; }
    BENCH_USER="$(id -un)"
    sudo -v || exit 1
    LOCAL="$REPO_ROOT/benchmark/results/local"
    RESUME="" ARGV=()
    for a in "$@"; do
        case "$a" in
            --resume) RESUME=latest ;;
            --resume=*) RESUME="${a#*=}" ;;
            *) ARGV+=("$a") ;;
        esac
    done
    if [ -n "$RESUME" ]; then
        # continue a session: it keeps the arguments of the first run, so the unit list matches
        if [ "$RESUME" = latest ]; then
            SESSION="$(ls -td "$LOCAL"/session_*/ 2>/dev/null | while read -r d; do [ -f "${d}quiet_args.txt" ] && { echo "${d%/}"; break; }; done)"
        else SESSION="$RESUME"; fi
        [ -f "${SESSION:-/nonexistent}/quiet_args.txt" ] || { echo "no session to resume (looked in $LOCAL/session_*)" >&2; exit 1; }
        [ "${#ARGV[@]}" -gt 0 ] && echo "note: arguments are taken from the session; ignoring: ${ARGV[*]}"
        mapfile -t ARGV <"$SESSION/quiet_args.txt"
        RESUME_FLAG=1
        echo "Resuming $SESSION"
        "$HERE/units_report.sh" "$SESSION"
    else
        SESSION="$LOCAL/session_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$SESSION"
        : >"$SESSION/quiet_args.txt"
        [ "${#ARGV[@]}" -gt 0 ] && printf '%s\n' "${ARGV[@]}" >"$SESSION/quiet_args.txt"
        RESUME_FLAG=0
    fi
    LOG="$LOCAL/quiet_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$LOCAL"
    echo "Closing KDE and all your processes in 10 s (Ctrl+C now to abort). Log: $LOG, session: $SESSION"
    echo "A status screen will appear on virtual console 9; q (then y) there aborts and returns to KDE."
    echo "Continue later with: $SELF --resume"
    sleep 10
    exec sudo systemd-run --unit=ristretto-quiet --collect --property=KillMode=control-group \
        /bin/bash "$SELF" --inner "$BENCH_USER" "$LOG" "$SESSION" "$RESUME_FLAG" "${ARGV[@]}"
fi

# ---- inner: root, transient system unit --------------------------------------------------------
shift
BENCH_USER="$1" BENCH_LOG="$2" SESSION="$3" RESUME_FLAG="$4"
shift 4
BENCH_UID="$(id -u "$BENCH_USER" 2>/dev/null)"
# Never touch processes by name or with a non-user id: a wrong user here once meant `pkill -u root`.
[ "$(id -u)" = 0 ] || { echo "### --inner is for the transient unit, which runs as root" >&2; exit 1; }
if [ -z "$BENCH_UID" ] || [ "$BENCH_UID" -lt 1000 ] || [ "$BENCH_UID" -ge 60000 ]; then
    echo "### refusing to run: '$BENCH_USER' (uid '${BENCH_UID:-?}') is not a regular user" >&2
    exit 1
fi
BENCH_HOME="$(getent passwd "$BENCH_USER" | cut -d: -f6)"
NO_NETWORK=0 SMT_MODE=auto VT=9 COOLDOWN=30 COOL_TEMP=50 THREADS_ARG="" ARGS=()
for a in "$@"; do
    case "$a" in
        --no-network) NO_NETWORK=1 ;;
        --smt-mode=*) SMT_MODE="${a#*=}" ;;
        --smt-off) SMT_MODE=off ;;
        --vt=*) VT="${a#*=}" ;;
        --threads=*) THREADS_ARG="${a#*=}" ;;
        --cooldown=*) COOLDOWN="${a#*=}" ;;
        --cooldown-temp=*) COOL_TEMP="${a#*=}" ;;
        *) ARGS+=("$a") ;;
    esac
done
case "$SMT_MODE" in auto|on|off) ;; *) echo "### --smt-mode must be auto, on or off" >&2; exit 1 ;; esac
exec >>"$BENCH_LOG" 2>&1
echo "### quiet_run start $(date -Is), user $BENCH_USER ($BENCH_UID), status VT $VT"

mkdir -p "$RUN"; rm -f "$RUN/abort"
echo "$BENCH_LOG" >"$RUN/log"; echo "$SESSION" >"$RUN/session"; date +%s >"$RUN/start"; echo "${ARGS[*]:-<defaults>}" >"$RUN/args"
state() { echo "$1" >"$RUN/state"; echo "### state: $1 ($(date +%T))"; }

CPU=/sys/devices/system/cpu
declare -A SAVED
ORDER=()
# keep the first value only: smt/control is set twice (off, then on), and the second save would
# overwrite the original "on" with "off", so the restore left SMT off
save() { [ -n "${SAVED[$1]+x}" ] && return; SAVED[$1]="$(cat "$1" 2>/dev/null)"; ORDER+=("$1"); }
setv() { [ -w "$1" ] && { save "$1"; echo "$2" >"$1" || echo "### could not set $1"; }; }
STOPPED=()
BENCH_PID=""

restore() {
    trap - EXIT INT TERM HUP
    echo "### restoring $(date -Is)"
    if [ -n "$BENCH_PID" ] && kill -0 "$BENCH_PID" 2>/dev/null; then
        kill -TERM -- "-$BENCH_PID" 2>/dev/null; sleep 3; kill -KILL -- "-$BENCH_PID" 2>/dev/null
    fi
    state "done: restoring settings, restarting KDE"
    # siblings first, so their cpufreq files exist again for the loop below
    if [ -n "${SAVED[$CPU/smt/control]+x}" ]; then echo "${SAVED[$CPU/smt/control]}" >"$CPU/smt/control" 2>/dev/null; sleep 1; fi
    for ((i = ${#ORDER[@]} - 1; i >= 0; i--)); do echo "${SAVED[${ORDER[$i]}]}" >"${ORDER[$i]}" 2>/dev/null; done
    for ((i = ${#STOPPED[@]} - 1; i >= 0; i--)); do systemctl start "${STOPPED[$i]}"; done
    systemctl stop ristretto-status 2>/dev/null
    systemctl start "$DM"
    echo "### restored, KDE restarting $(date -Is)"
}
trap restore EXIT INT TERM HUP

# 1. desktop and background noise ---------------------------------------------------------------
state "stopping desktop"
stop_unit() { systemctl is-active --quiet "$1" && { systemctl stop "$1" && STOPPED+=("$1"); }; }
systemctl is-active --quiet "$DM" || echo "### $DM was not active; it will be started at the end anyway"
systemctl stop "$DM"
# status screen on its own VT (above NAutoVTs, so no getty competes for the keyboard)
systemd-run --unit=ristretto-status --collect \
    --property=StandardInput=tty-force --property=StandardOutput=tty --property=StandardError=tty \
    --property=TTYPath="/dev/tty$VT" --property=TTYReset=yes --property=TTYVHangup=yes --property=TTYVTDisallocate=yes \
    /bin/bash "$HERE/quiet_status.sh"
chvt "$VT"
# anything the session left behind (user manager, Baloo, browsers, ...): end all sessions of the user
loginctl terminate-user "$BENCH_USER"
sleep 2
pkill -KILL -U "$BENCH_UID"
for s in "${NOISY[@]}"; do stop_unit "$s"; done
[ "$NO_NETWORK" = 1 ] && for s in "${NETWORK[@]}"; do stop_unit "$s"; done
systemctl is-active --quiet "user@$BENCH_UID" && systemctl stop "user@$BENCH_UID"
sync; echo 3 >/proc/sys/vm/drop_caches
echo "### stopped: ${STOPPED[*]}"

# 2. performance profile --------------------------------------------------------------------------
state "setting performance profile"
perf_profile() {   # re-run after every SMT switch: CPUs that come online get default cpufreq settings
    local c max
    for c in "$CPU"/cpu[0-9]*/cpufreq; do
        [ -d "$c" ] || continue
        # an offline CPU keeps a stale cpufreq directory whose files answer "Device or resource busy"
        [ "$(cat "${c%/cpufreq}/online" 2>/dev/null)" = 0 ] && continue
        setv "$c/scaling_governor" performance
        max="$(cat "$c/cpuinfo_max_freq")"
        # no_turbo=1 leaves the highest non-turbo step as the ceiling; the kernel ignores a too-high min
        setv "$c/scaling_max_freq" "$max"
        setv "$c/scaling_min_freq" "$max"
    done
}
smt_set() {   # on|off
    [ -w "$CPU/smt/control" ] || return 0
    [ "$(cat "$CPU/smt/control")" = "$1" ] && return 0
    setv "$CPU/smt/control" "$1"
    sleep 1
    # offlining a CPU drops it from the affinity of every task, onlining it again does not give it
    # back: without this the group after "off" inherits a mask without the siblings (numactl: "cpu
    # argument 1,3 is out of range")
    taskset -pc "$(cat "$CPU/online")" $$ >/dev/null 2>&1 || true
    perf_profile
}
smt_set on   # known starting point: every sibling online, so the topology below is complete
perf_profile
setv "$CPU/intel_pstate/no_turbo" 1
[ -w "$CPU/intel_pstate/min_perf_pct" ] && setv "$CPU/intel_pstate/min_perf_pct" 100
setv /proc/sys/kernel/nmi_watchdog 0
setv /proc/sys/kernel/numa_balancing 0
setv /proc/sys/vm/swappiness 1
setv /sys/kernel/mm/transparent_hugepage/enabled madvise
N_PHYS="$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)"
N_LOGICAL="$(lscpu -p=CPU | grep -v '^#' | wc -l)"
echo "### governor: $(cat $CPU/cpu0/cpufreq/scaling_governor), freq: $(cat $CPU/cpu0/cpufreq/scaling_cur_freq) kHz, smt: $(cat $CPU/smt/control), cores: $N_PHYS, logical: $N_LOGICAL"

# 3. thread counts -> SMT groups --------------------------------------------------------------------
# Counts up to the number of physical cores run with SMT off (the siblings cannot disturb a pinned
# thread), counts above it need the siblings and run with SMT on and `run_comparison.sh --smt`.
# --smt-mode=on keeps SMT on for every count; off drops the counts above the physical cores.
if [ -n "$THREADS_ARG" ]; then IFS=, read -ra TLIST <<<"$THREADS_ARG"
else
    TLIST=()
    for ((t = 1; t <= N_LOGICAL; t *= 2)); do TLIST+=("$t"); done
    [ "${TLIST[-1]}" != "$N_LOGICAL" ] && TLIST+=("$N_LOGICAL")
fi
LOW=() HIGH=()
for t in "${TLIST[@]}"; do
    if [ "$t" -le "$N_PHYS" ]; then LOW+=("$t")
    elif [ "$SMT_MODE" = off ]; then echo "### dropping $t threads: more than the $N_PHYS physical cores and --smt-mode=off"
    else HIGH+=("$t"); fi
done
RESUME_ARG=(); [ "$RESUME_FLAG" = 1 ] && RESUME_ARG=(--resume)
# Write every group's units into the session's unit list up front, so the status screen can show what
# is left; a group with nothing left to run (everything done in an earlier run) is skipped.
plan_group() {   # low|high -> status 0 when units remain
    local list flags=()
    if [ "$1" = low ]; then list="$(IFS=,; echo "${LOW[*]}")"; else list="$(IFS=,; echo "${HIGH[*]}")"; flags=(--smt); fi
    runuser -u "$BENCH_USER" -- env HOME="$BENCH_HOME" "$REPO_ROOT/benchmark/local/run_comparison.sh" --plan-only \
        --threads="$list" --session="$SESSION" "${flags[@]}" "${ARGS[@]}" "${RESUME_ARG[@]}" >/dev/null 2>&1
    [ $? -ne 10 ]
}
GROUPS_TODO=()
[ "${#LOW[@]}" -gt 0 ] && plan_group low && GROUPS_TODO+=("low")
[ "${#HIGH[@]}" -gt 0 ] && plan_group high && GROUPS_TODO+=("high")
echo "### session $SESSION (resume: $RESUME_FLAG)"
echo "### smt mode $SMT_MODE; physical-core counts: ${LOW[*]:-none}; SMT counts: ${HIGH[*]:-none}"

max_temp() { local f m=0 t; for f in /sys/class/hwmon/hwmon*/temp*_input; do
    t="$(cat "$f" 2>/dev/null)"; [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -gt "$m" ] && m=$t; done; echo "$((m / 1000))"; }

run_group() {   # group name: low|high
    local list flags=() want
    if [ "$1" = low ]; then
        list="$(IFS=,; echo "${LOW[*]}")"
        [ "$SMT_MODE" = on ] && want=on || want=off
    else
        list="$(IFS=,; echo "${HIGH[*]}")"; want=on; flags=(--smt)
    fi
    smt_set "$want"
    state "running benchmark: $list thread(s), SMT $want"
    echo "### starting benchmark group $1 (threads $list, smt $(cat "$CPU/smt/control")) $(date -Is)"
    # own session/process group, so an abort can kill the whole tree (julia, bart, numactl) at once
    setsid nice -n -5 runuser -u "$BENCH_USER" -- env HOME="$BENCH_HOME" \
        "$REPO_ROOT/benchmark/local/run_comparison.sh" --threads="$list" --cooldown="$COOLDOWN" \
        --cooldown-temp="$COOL_TEMP" --session="$SESSION" "${flags[@]}" "${ARGS[@]}" "${RESUME_ARG[@]}" &
    BENCH_PID=$!
    while kill -0 "$BENCH_PID" 2>/dev/null; do
        if [ -e "$RUN/abort" ]; then
            state "aborting benchmark"
            kill -TERM -- "-$BENCH_PID" 2>/dev/null; sleep 3; kill -KILL -- "-$BENCH_PID" 2>/dev/null
            pkill -KILL -U "$BENCH_UID"
            break
        fi
        sleep 1
    done
    wait "$BENCH_PID" 2>/dev/null
    echo "### group $1 exit status $?"
}

first=1
for g in "${GROUPS_TODO[@]}"; do
    [ -e "$RUN/abort" ] && break
    if [ "$first" = 0 ]; then   # cooldown between groups only, as run_comparison.sh does between runs
        state "cooldown between groups"
        for _ in $(seq "$COOLDOWN"); do [ -e "$RUN/abort" ] && break; sleep 1; done
        waited=0
        while [ "$(max_temp)" -gt "$COOL_TEMP" ] && [ "$waited" -lt 900 ] && [ ! -e "$RUN/abort" ]; do sleep 5; waited=$((waited + 5)); done
    fi
    first=0
    run_group "$g"
done
[ -e "$RUN/abort" ] && state "aborted" || state "done"
# restore() runs on exit

#!/bin/bash
# Status screen of quiet_run.sh, run by systemd on a spare virtual console (default VT 9). Refreshes
# every 2 s. Keys:  q  abort the benchmark and return to the desktop (asks y to confirm)
# State comes from /run/ristretto-quiet/{state,log,abort}, written by quiet_run.sh.
RUN="${RUN:-/run/ristretto-quiet}"
INTERVAL=2
abort() { touch "$RUN/abort"; }
trap 'abort' INT
trap 'exit 0' TERM

# Every sysfs number is checked before it enters arithmetic: a file can read empty (CPU idling or
# going offline), which used to give "syntax error: operand expected" on the screen.
num() { local v; v="$(cat "$1" 2>/dev/null)"; [[ "$v" =~ ^[0-9]+$ ]] && echo "$v"; }
freq() { local f v s=0 n=0; for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do
    v="$(num "$f")"; [ -n "$v" ] && { s=$((s + v)); n=$((n + 1)); }; done
    [ "$n" -gt 0 ] && echo "$((s / n / 1000)) MHz avg over $n CPUs" || echo "n/a"; }
temp() { local f m=0 t; for f in /sys/class/hwmon/hwmon*/temp*_input; do
    t="$(cat "$f" 2>/dev/null)"
    [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -gt "$m" ] && m=$t; done; echo "$((m / 1000)) C"; }
throttle() { cat /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count 2>/dev/null | awk '{s+=$1} END {print s+0}'; }

confirm=0
while true; do
    state="$(cat "$RUN/state" 2>/dev/null)"
    start="$(num "$RUN/start")"; start="${start:-$(date +%s)}"
    printf '\033[H\033[2J'
    echo "=== Ristretto quiet benchmark ===   $(date +%T)   elapsed $(( ($(date +%s) - start) / 60 )) min"
    echo "state     : ${state:-?}"
    echo "frequency : $(freq)    temp: $(temp)    throttle events: $(throttle)"
    echo "load      : $(cut -d' ' -f1-3 /proc/loadavg)    mem: $(free -m | awk '/Mem:/ {print $3 " / " $2 " MB used"}')"
    echo "args      : $(cat "$RUN/args" 2>/dev/null)"
    echo "---"
    cols=$(tput cols 2>/dev/null || echo 80)
    session="$(cat "$RUN/session" 2>/dev/null)"
    [ -n "$session" ] && "$(dirname "$(readlink -f "$0")")/units_report.sh" "$session" brief 2>/dev/null | cut -c1-"$cols"
    echo "---"
    runlog="$(ls -t /home/*/Ristretto.jl/benchmark/results/local/2*/*.log 2>/dev/null | head -n 1)"
    [ -n "$runlog" ] && { echo "$runlog"; tr '\r' '\n' <"$runlog" | grep -v '^\s*$' | tail -n 6 | cut -c1-"$cols"; }
    echo "---"
    tail -n 4 "$(cat "$RUN/log" 2>/dev/null)" 2>/dev/null | cut -c1-"$cols"
    echo
    case "$state" in
        done*|aborted*) echo "Finished: the desktop is coming back." ;;
        *) if [ "$confirm" = 1 ]; then echo ">>> ABORT the benchmark and restart KDE?  y = yes, any other key = no"
           else echo "[q] abort benchmark and return to the desktop      (Ctrl+Alt+F1/F2 does not stop anything)"; fi ;;
    esac
    if read -rsn1 -t "$INTERVAL" key; then
        if [ "$confirm" = 1 ]; then [ "$key" = y ] && abort; confirm=0
        elif [ "$key" = q ]; then confirm=1; fi
    fi
    [ -e "$RUN/abort" ] && echo "abort requested..." 
done

#!/bin/bash
# Progress report of a measurement session: which (threads, backend, BART mode, section) units are
# done and how long each took, which are left and how long they should take.
#   units_report.sh SESSION_DIR [full|brief]
# Reads SESSION_DIR/units.tsv (unit<TAB>status<TAB>seconds<TAB>finished), written by run_comparison.sh.
# The estimate for a unit that has not run is the mean time of the done units of the same section
# (any thread count), else the mean of all done units; units of a section never seen are not counted.
f="$1/units.tsv"; mode="${2:-full}"
[ -s "$f" ] || { echo "no unit list in $1"; exit 0; }
awk -F'\t' -v mode="$mode" '
function hms(s) { s = int(s + 0.5); return sprintf("%d:%02d:%02d", int(s / 3600), int((s % 3600) / 60), s % 60) }
function pretty(u,   p) { split(u, p, ":"); return sprintf("%s thread%s, %s, bart %s, %s", p[1], p[1] == 1 ? "" : "s", p[2], p[3], p[4]) }
{
    n++; unit[n] = $1; st[n] = $2; sec[n] = $3; fin[n] = $4
    split($1, p, ":")
    if ($2 == "done") { d++; tot += $3; ssum[p[4]] += $3; scnt[p[4]]++ }
    else if ($2 == "failed") fl++
    else if ($2 == "running") r++
    else pend++
}
END {
    for (i = 1; i <= n; i++) if (st[i] != "done" && st[i] != "failed") {
        split(unit[i], p, ":")
        if (scnt[p[4]] > 0) { e = ssum[p[4]] / scnt[p[4]]; est += e; estof[i] = e }
        else if (d > 0) { e = tot / d; est += e; estof[i] = e; guess = 1 }
        else unk++
    }
    printf "Progress: %d/%d done, %d failed, %d running, %d pending.  Time in done units: %s", d, n, fl, r, pend, hms(tot)
    if (pend + r > 0) {
        if (d > 0) printf ".  Estimated remaining: %s%s", hms(est), guess ? " (rough: some sections not measured yet)" : ""
        else printf ".  No unit has finished yet, so no estimate"
    }
    printf "\n"
    shown = 0
    for (i = 1; i <= n; i++) {
        if (mode == "brief") {
            if (st[i] == "done") { dn[++nd] = i; continue }
            if (st[i] != "running" && ++np > 5) continue
        }
        if (st[i] == "done") line = sprintf("  [x] %s  %s  (finished %s)", hms(sec[i]), pretty(unit[i]), fin[i])
        else if (st[i] == "failed") line = sprintf("  [!] failed after %s  %s  (finished %s)", hms(sec[i]), pretty(unit[i]), fin[i])
        else if (st[i] == "running") line = sprintf("  [>] running      %s", pretty(unit[i]))
        else line = sprintf("  [ ] pending%s  %s", (i in estof) ? sprintf(" ~%s", hms(estof[i])) : "", pretty(unit[i]))
        if (mode == "brief") brief[i] = line; else print line
    }
    if (mode == "brief") {
        for (k = (nd > 4 ? nd - 3 : 1); k <= nd; k++) printf "  [x] %s  %s\n", hms(sec[dn[k]]), pretty(unit[dn[k]])
        for (i = 1; i <= n; i++) if (i in brief) print brief[i]
        if (np > 5) printf "  ... and %d more pending\n", np - 5
    }
    exit (pend + r + fl > 0) ? 10 : 0
}' "$f"

#!/bin/bash
# NUMA-isolated concurrent benchmark matrix. Books one node --exclusive, then runs several
# configurations of the matrix AT THE SAME TIME inside that one job, each pinned via `numactl` to
# its own disjoint NUMA domain (no two tasks ever share cores, L3, or memory channels). This is the
# fix for the NUMA-fragmentation anomaly: a task scattered across domains by a busy shared-node
# scheduler saw BART/OpenBLAS/FFTW timings inflate 10-700x.
#
# The node topology is discovered with ThreadPinning.jl, never hardcoded. Each domain must have at
# least 16 cores (the largest thread count in the default matrix).
#
# It runs benchmark/run.jl (Ristretto only, every catalog case and method). The cross-toolkit
# comparison has its own copy of this script in RistrettoBenchmarking
# (https://github.com/hakkelt/RistrettoBenchmarking).
#
# Matrix dimensions (each a comma-separated list):
#   --matrix-threads=1,2,4,8,16          thread counts (default: 1,2,4,8,16)
#   --matrix-backends=openblas,mkl       BLAS backends (default: both)
#   --matrix-refs=master:/path,perf:/path   Ristretto checkouts to measure, each passed to
#                                        run.jl as --ristretto=<path> --ref-name=<name>. Default: this
#                                        checkout. A third field redirects dev'd packages of that
#                                        ref (run.jl --dev), `+` joining several:
#                                        perf:/path:NestedThreading=/nt+OperatorCore=/oc
#   --swap-repeat                        queue the whole matrix a second time (with --remeasure),
#                                        refs and environment variants in reverse order.
#                                        Neighbouring variants start on neighbouring domains, the
#                                        first always on the lower one, and domains differed by up
#                                        to 1.6x on a 2 ms case; the repeat puts each variant on
#                                        another domain. Compare with `compare.jl --pick=min`.
#   --matrix-env=KMP_BLOCKTIME=0,KMP_BLOCKTIME=200   environment variants; `+` joins several
#                                        assignments into one variant (A=1+B=2). Default: none.
#
# Placement:
#   (default)          isolated: one task per NUMA domain, whatever its thread count. Timings are
#                      free of any interference between tasks; use this for published numbers and
#                      whenever memory-bandwidth contention is itself the question.
#   --pack             several tasks may share a domain, each on its own physical cores (never an
#                      SMT sibling, never a core another task holds) and with its memory still bound
#                      to that domain. Cores are taken best-fit, from the fullest L3 group and then
#                      the fullest domain that still fits, so small tasks fill up the same L3 and
#                      domain and leave the others free for large ones. Packed tasks share L3 and
#                      memory bandwidth with their neighbours, so their results get a node class of
#                      their own (RISTRETTO_BENCH_PLACEMENT=shared, see node_class in utils/harness.jl) and
#                      are never reused for, or silently compared with, an isolated run.
#   --pack-mem-gb=G    memory reserved per packed task (default 6; the harness's tasks peak at
#                      4.5 GB). A task starts on a domain only while
#                      the reservations there stay within 90% of its MemTotal.
# Everything else (--cases=, --methods=, --remeasure, ...) is passed
# verbatim to every launched Julia process.
#
# usage (through submit.sh, which picks the partition):
#   benchmark/slurm/submit.sh matrix.sh --matrix-refs=master:/wt/master,perf:/wt/perf
#
#SBATCH --job-name=matrix
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
set -u

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_ROOT/benchmark/slurm/common.sh"

### ---- 1. Parse flags: matrix-level vs pass-through --------------------------------------------

MATRIX_THREADS=(1 2 4 8 16)
MATRIX_BACKENDS=(openblas mkl)
MATRIX_REFS=("")
MATRIX_ENV=("")
PACK=0
PACK_MEM_GB=""
SWAP_REPEAT=0
PASS_ARGS=()
for a in "$@"; do
    case "$a" in
        --pack) PACK=1 ;;
        --swap-repeat) SWAP_REPEAT=1 ;;
        --pack-mem-gb=*) PACK_MEM_GB="${a#*=}" ;;
        --matrix-threads=*) IFS=, read -ra MATRIX_THREADS <<<"${a#*=}" ;;
        --matrix-backends=*) IFS=, read -ra MATRIX_BACKENDS <<<"${a#*=}" ;;
        --matrix-refs=*) IFS=, read -ra MATRIX_REFS <<<"${a#*=}" ;;
        --matrix-env=*) IFS=, read -ra MATRIX_ENV <<<"${a#*=}" ;;
        *) PASS_ARGS+=("$a") ;;
    esac
done

PROJECT_DIR="benchmark"
SCRIPT_PATH="benchmark/run.jl"
RESULTS_DIR="benchmark/results/runs"
: "${PACK_MEM_GB:=6}"
LOG_DIR="$BENCH_RESULTS/slurm/matrix_${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
START_MARKER="$LOG_DIR/.start"
touch "$START_MARKER"

### ---- 2. NUMA topology discovery (ThreadPinning.jl, never hardcoded) ---------------------------

# Only one CPU thread per physical core is listed: a task is never given an SMT sibling, whose core
# it would share with another of its own threads.
mapfile -t NUMA_CORES < <("$JULIA_BIN" --project="$PROJECT_DIR" -e '
    using ThreadPinning
    for i in 1:ThreadPinning.nnuma()
        println(join(filter(!ThreadPinning.ishyperthread, ThreadPinning.numa(i)), " "))
    end')
N_DOMAINS=${#NUMA_CORES[@]}
if [ "$N_DOMAINS" -lt 1 ]; then
    echo "### NUMA topology discovery failed (no domains reported)" >&2
    exit 1
fi
MAX_THREADS=$(printf '%s\n' "${MATRIX_THREADS[@]}" | sort -n | tail -1)
for d in "${!NUMA_CORES[@]}"; do
    n=$(wc -w <<<"${NUMA_CORES[$d]}")
    if [ "$n" -lt "$MAX_THREADS" ]; then
        echo "### domain $d only has $n cores (< $MAX_THREADS threads requested), aborting" >&2
        exit 1
    fi
done
echo "### discovered $N_DOMAINS NUMA domains, $(wc -w <<<"${NUMA_CORES[0]}") physical cores each"

# Which L3 each core belongs to (the kernel's shared_cpu_list of its last-level cache, used only as
# a grouping key) and how much memory each domain has, for --pack.
declare -A L3_OF
declare -a DOMAIN_MEM_MB DOMAIN_MEM_USED_MB
for d in "${!NUMA_CORES[@]}"; do
    for c in ${NUMA_CORES[$d]}; do
        L3_OF[$c]="$(cat "/sys/devices/system/cpu/cpu$c/cache/index3/shared_cpu_list" 2>/dev/null || echo "domain$d")"
    done
    kb=$(awk '/MemTotal/ {print $4}' "/sys/devices/system/node/node$d/meminfo" 2>/dev/null)
    DOMAIN_MEM_MB[d]=$(( ${kb:-0} / 1024 ))
    DOMAIN_MEM_USED_MB[d]=0
done
PACK_MEM_MB=$(( PACK_MEM_GB * 1024 ))
if [ "$PACK" = 1 ]; then
    for d in "${!NUMA_CORES[@]}"; do
        if [ $(( DOMAIN_MEM_MB[d] * 9 / 10 )) -lt "$PACK_MEM_MB" ]; then
            echo "### --pack: domain $d has ${DOMAIN_MEM_MB[$d]} MB, less than one ${PACK_MEM_GB} GB reservation, aborting" >&2
            exit 1
        fi
    done
    echo "### --pack: tasks share domains on disjoint cores, ${PACK_MEM_GB} GB reserved per task, results tagged RISTRETTO_BENCH_PLACEMENT=shared"
    # Memory-bound cases lose most to packing. Measured on the same code, packed / isolated
    # (EPYC 7763, 20 tasks on 8 domains, 2026-09-24): 3D 1.42x geomean (up to 3.2x), cine 1.24x,
    # multislice 1.18x (up to 4.0x), against 1.03-1.07x for the 2D and radial cases.
    heavy_ids="shepp_logan_3d_8ch_cartesian shepp_logan_multislice_8ch_cartesian torso_cine_8ch_cartesian torso_cine_8ch_radial"
    real_heavy_ids="real_3d_multichannel_cartesian real_multislice_multichannel_cartesian real_cine_multichannel_cartesian real_cine_multichannel_radial"
    case_patterns=""
    [ "${RISTRETTO_BENCH_REAL_DATA:-}" = 1 ] && heavy_ids="$heavy_ids $real_heavy_ids"
    for a in "${PASS_ARGS[@]}"; do
        case "$a" in
            --cases=*) case_patterns="${a#*=}" ;;
            --real | --data=real | --data=all) heavy_ids="$heavy_ids $real_heavy_ids" ;;
        esac
    done
    heavy_selected=()
    for id in $heavy_ids; do
        if [ -z "$case_patterns" ]; then
            heavy_selected+=("$id")
            continue
        fi
        IFS=, read -ra pats <<<"$case_patterns"
        for p in "${pats[@]}"; do
            [[ "$id" == *"${p,,}"* ]] && { heavy_selected+=("$id"); break; }
        done
    done
    if [ "${#heavy_selected[@]}" -gt 0 ]; then
        echo "### WARNING: --pack with memory-bound cases (${heavy_selected[*]}): packing slowed them" \
            "1.2-1.4x on average and up to 4x in measurement, unevenly across tasks. Use these timings" \
            "for smoke or functional checks only; time these cases without --pack." >&2
    fi
    echo "### L3 groups: $(printf '%s\n' "${L3_OF[@]}" | sort -u | wc -l), domain memory: ${DOMAIN_MEM_MB[*]} MB"
fi

# Refs and environment variants vary fastest, so the variants of one (threads, backend) pair start
# together on neighbouring domains and see the same node state: a comparison between them is not
# confounded by drift over the hours the job runs.
QUEUE=()
for t in "${MATRIX_THREADS[@]}"; do
    for b in "${MATRIX_BACKENDS[@]}"; do
        for e in "${!MATRIX_ENV[@]}"; do
            for r in "${!MATRIX_REFS[@]}"; do
                QUEUE+=("$t:$b:$r:$e:0")
            done
        done
    done
done
# The repeat passes --remeasure: otherwise it would find the first pass's results stored and skip.
if [ "$SWAP_REPEAT" = 1 ]; then
    for t in "${MATRIX_THREADS[@]}"; do
        for b in "${MATRIX_BACKENDS[@]}"; do
            for ((e = ${#MATRIX_ENV[@]} - 1; e >= 0; e--)); do
                for ((r = ${#MATRIX_REFS[@]} - 1; r >= 0; r--)); do
                    QUEUE+=("$t:$b:$r:$e:1")
                done
            done
        done
    done
fi
echo "### queue: ${QUEUE[*]}"
echo "### refs: ${MATRIX_REFS[*]:-<this checkout>}  env variants: ${MATRIX_ENV[*]:-<none>}"
echo "### pass-through args: ${PASS_ARGS[*]:-<none>}"

### ---- 3. Scheduler: tasks on disjoint cores of one NUMA domain, enforced via numactl -----------
#
# Isolated (default): a task needs a domain nobody else holds. --pack: a task needs `threads` free
# cores of one domain plus a memory reservation there. Either way no core is ever held twice.

declare -A CORE_BUSY    # core id -> 1 while a task holds it
declare -A PID_DOMAIN   # background pid -> domain id
declare -A PID_CORES    # background pid -> its cores, space separated

free_cores() {  # domain -> its free cores, space separated
    local c out=()
    for c in ${NUMA_CORES[$1]}; do [ -z "${CORE_BUSY[$c]:-}" ] && out+=("$c"); done
    echo "${out[*]}"
}

pick_cores() {  # domain threads -> the cores to give the task (empty: it does not fit there)
    local d=$1 t=$2 free n
    free=($(free_cores "$d"))
    n=${#free[@]}
    if [ "$PACK" = 0 ]; then
        [ "$n" = "$(wc -w <<<"${NUMA_CORES[$d]}")" ] && echo "${free[*]:0:t}"
        return
    fi
    [ "$n" -lt "$t" ] && return
    [ $(( DOMAIN_MEM_USED_MB[d] + PACK_MEM_MB )) -gt $(( DOMAIN_MEM_MB[d] * 9 / 10 )) ] && return
    # Best fit over L3 groups: the group with the fewest free cores that still holds all t threads.
    local key best="" best_n=1000000 c
    declare -A group_n=()
    for c in "${free[@]}"; do key=${L3_OF[$c]}; group_n[$key]=$(( ${group_n[$key]:-0} + 1 )); done
    for key in "${!group_n[@]}"; do
        if [ "${group_n[$key]}" -ge "$t" ] && [ "${group_n[$key]}" -lt "$best_n" ]; then
            best=$key; best_n=${group_n[$key]}
        fi
    done
    local out=()
    if [ -n "$best" ]; then
        for c in "${free[@]}"; do [ "${L3_OF[$c]}" = "$best" ] && out+=("$c"); done
    else
        out=("${free[@]}")   # no single L3 fits: spans several, but stays within the domain
    fi
    echo "${out[*]:0:t}"
}

pick_domain() {  # threads -> "domain cores..." (empty: nothing fits now)
    local t=$1 d cores best="" best_free=1000000 nfree
    for ((d = 0; d < N_DOMAINS; d++)); do
        cores=$(pick_cores "$d" "$t")
        [ -z "$cores" ] && continue
        [ "$PACK" = 0 ] && { echo "$d $cores"; return; }
        nfree=$(wc -w <<<"$(free_cores "$d")")
        [ "$nfree" -lt "$best_free" ] && { best="$d $cores"; best_free=$nfree; }
    done
    echo "$best"
}

reap_finished() {
    local pid c
    for pid in "${!PID_DOMAIN[@]}"; do
        kill -0 "$pid" 2>/dev/null && continue
        for c in ${PID_CORES[$pid]}; do unset "CORE_BUSY[$c]"; done
        [ "$PACK" = 1 ] && DOMAIN_MEM_USED_MB[${PID_DOMAIN[$pid]}]=$(( DOMAIN_MEM_USED_MB[${PID_DOMAIN[$pid]}] - PACK_MEM_MB ))
        unset "PID_DOMAIN[$pid]" "PID_CORES[$pid]"
    done
}

launch_task() {  # threads backend ref_index env_index repeat domain cores...
    local threads=$1 backend=$2 r=$3 e=$4 rep=$5 domain=$6
    shift 6
    local cores=("$@")
    local pin
    pin="$(tr ' ' ',' <<<"${cores[*]}")"
    local args=(--threads="$threads")
    [ "$backend" = mkl ] && args+=(--use-mkl)
    [ "$rep" = 1 ] && args+=(--remeasure)
    local tag="${backend}_${threads}t"
    [ "$rep" = 1 ] && tag="${tag}_repeat"
    local ref="${MATRIX_REFS[$r]}"
    if [ -n "$ref" ]; then
        local ref_name ref_path ref_dev
        IFS=: read -r ref_name ref_path ref_dev <<<"$ref"
        args+=(--ref-name="$ref_name" --ristretto="$ref_path")
        [ -n "$ref_dev" ] && args+=(--dev="${ref_dev//+/,}")
        tag="${ref_name}_$tag"
    fi
    local envs=()
    if [ -n "${MATRIX_ENV[$e]}" ]; then
        IFS=+ read -ra envs <<<"${MATRIX_ENV[$e]}"
        tag="${tag}_env$e"
        args+=(--env-variant="${MATRIX_ENV[$e]}")
    fi
    local log="$LOG_DIR/${tag}_domain${domain}.log"
    local placement=isolated
    [ "$PACK" = 1 ] && placement=shared

    # A plain background process, not an `srun` step: this SLURM build's step allocator picks the
    # step's cpu set itself (always starting from the lowest free id) before looking at --cpu-bind,
    # and rejects a mask outside what it picked, so concurrent --overlap steps can never receive
    # disjoint, caller-chosen cpu ranges. The job owns the whole node (--exclusive), so a background
    # process inherits the full cpuset and numactl enforces the placement. Without steps there is
    # no per-task `sacct` row; `/usr/bin/time -v` gives the per-task efficiency instead (see
    # report_efficiency.sh).
    numactl --physcpubind="$pin" --membind="$domain" \
        /usr/bin/time -v \
        env JULIA_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" \
            MKL_NUM_THREADS="$threads" RISTRETTO_BENCH_PLACEMENT="$placement" "${envs[@]}" \
        "$JULIA_BIN" --project="$PROJECT_DIR" -t "$threads" "$SCRIPT_PATH" \
            "${args[@]}" "${PASS_ARGS[@]}" \
        >"$log" 2>&1 &
    PID_DOMAIN[$!]=$domain
    PID_CORES[$!]="${cores[*]}"
    local c
    for c in "${cores[@]}"; do CORE_BUSY[$c]=1; done
    [ "$PACK" = 1 ] && DOMAIN_MEM_USED_MB[domain]=$(( DOMAIN_MEM_USED_MB[domain] + PACK_MEM_MB ))
    echo "### launched $tag on domain $domain (cores $pin, $placement), pid $! -> $log"
}

for cfg in "${QUEUE[@]}"; do
    IFS=: read -r threads backend r e rep <<<"$cfg"
    while true; do
        slot=$(pick_domain "$threads")
        [ -n "$slot" ] && break
        wait -n
        reap_finished
    done
    # shellcheck disable=SC2086  # "domain core core ..." splits into the positional arguments
    launch_task "$threads" "$backend" "$r" "$e" "$rep" $slot
done
wait

### ---- 4. Post-run sanity check: no run this job wrote spans more than one domain --------------

echo "### post-run NUMA sanity check"
find "$RESULTS_DIR" -name '*.json' -newer "$START_MARKER" | "$JULIA_BIN" --project="$PROJECT_DIR" -e '
    using JSON
    for f in eachline(stdin)
        d = try JSON.parsefile(f) catch; continue end
        pinned = get(d, "pinned_cpus", "")
        isempty(pinned) && continue
        ids = sort(parse.(Int, split(pinned, ",")))
        span = ids[end] - ids[1]
        println(basename(f), ": ", span < 16 ? "PASS" : "FAIL (span=$span, cpus=$pinned)")
    end
'

echo "### matrix.sh done $(date '+%H:%M:%S'); per-task logs in $LOG_DIR"

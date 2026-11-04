#!/bin/bash
#SBATCH --job-name=ristretto-large-datasets
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#
# Ristretto on the two full-size datasets of benchmark/large_datasets/run.jl, on the host: one run per
# thread count, all at the same time, each pinned to the physical cores of its own NUMA domain. Its
# memory is preferred on that domain but may spill to the others (the 3D volume needs more than
# one domain holds).
#
#   benchmark/slurm/submit.sh --production large_datasets.sh [--threads=1,4,8] [run.jl args...]
#   benchmark/slurm/submit.sh large_datasets.sh --threads=16 --prepare
#
# `--prepare` builds and caches the cases only; run it once before the timed runs.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

THREADS="1,4,8"
PASS=()
for a in "$@"; do
    case "$a" in
        --threads=*) THREADS="${a#*=}" ;;
        *) PASS+=("$a") ;;
    esac
done
IFS=, read -ra THREAD_LIST <<<"$THREADS"
PROJECT_DIR=benchmark
LOG_DIR="$BENCH_RESULTS/slurm/large_${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"

mapfile -t NUMA_CORES < <("$JULIA_BIN" --project="$PROJECT_DIR" -e '
    using ThreadPinning
    for i in 1:ThreadPinning.nnuma()
        println(join(filter(!ThreadPinning.ishyperthread, ThreadPinning.numa(i)), " "))
    end')
[ "${#NUMA_CORES[@]}" -ge "${#THREAD_LIST[@]}" ] || { echo "### ${#NUMA_CORES[@]} NUMA domains for ${#THREAD_LIST[@]} runs" >&2; exit 1; }

for i in "${!THREAD_LIST[@]}"; do
    t="${THREAD_LIST[$i]}"
    read -ra cores <<<"${NUMA_CORES[$i]}"
    [ "${#cores[@]}" -ge "$t" ] || { echo "### domain $i has ${#cores[@]} cores, $t threads requested" >&2; exit 1; }
    pin=$(IFS=,; echo "${cores[*]:0:$t}")
    log="$LOG_DIR/${t}threads.log"
    numactl --physcpubind="$pin" --preferred="$i" /usr/bin/time -v \
        env OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" \
        "$JULIA_BIN" --project="$PROJECT_DIR" -t "$t" benchmark/large_datasets/run.jl "${PASS[@]}" \
        >"$log" 2>&1 &
    echo "### $t threads on domain $i (cores $pin), pid $! -> $log"
    # The cases are prepared by the first run that needs them; let it finish loading before the
    # others read the cache.
    [ "$i" -eq 0 ] && sleep 30
done
wait

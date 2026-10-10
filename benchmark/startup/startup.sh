#!/bin/bash
#SBATCH --job-name=ristretto-startup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Startup cost of every toolkit of the comparison (benchmark/startup/run.jl): load time, first vs
# warm solve and end-to-end fresh-process time, at 8 threads on 8 physical cores of one NUMA domain.
# Like load_time.sh, it books 16 cores rather than a whole node: the measurement is dominated by
# compilation and file reads, and the children run on 8 of them.
#
#   benchmark/slurm/submit.sh benchmark/startup/startup.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

export STARTUP_THREADS="${STARTUP_THREADS:-8}"
"$JULIA_BIN" --startup-file=no --project="$REPO_ROOT/benchmark/comparison" "$REPO_ROOT/benchmark/startup/run.jl"

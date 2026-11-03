#!/bin/bash
#SBATCH --job-name=ristretto-large-datasets-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Ristretto on the two full-size datasets of benchmark/large_datasets/run.jl, on one GPU (one host
# thread). Run benchmark/slurm/large_datasets.sh --prepare first, so the cases are cached.
#
#   benchmark/slurm/submit.sh --production large_datasets_gpu.sh [run.jl args...]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/benchmark/slurm/common.sh"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$JULIA_BIN" --project=benchmark -t 1 benchmark/large_datasets/run.jl --device=cuda "$@"

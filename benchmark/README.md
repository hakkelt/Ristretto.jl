# benchmark/

Performance and accuracy measurement of Ristretto. Everything here reads the
same **case catalog** (`utils/`), so two measurements of the same case, by two checkouts of Ristretto or by
Ristretto and another toolkit, reconstruct byte-identical inputs. The cross-toolkit comparison
(Ristretto against BART, SigPy, MRIReco, MIRT and MRpro) lives in
[RistrettoBenchmarking](https://github.com/hakkelt/RistrettoBenchmarking), which pins a commit of
this repository as a submodule and reads this catalog from it.

| path | what it is |
|---|---|
| `utils/` | the case catalog (`BenchUtils`): phantoms, sampling, noise, real-data loaders, Ristretto method specs, timing, result store |
| `run.jl` | the Ristretto harness: every catalog case × every applicable method, one checkout of Ristretto at a time |
| `compare.jl` | compares two checkouts (refs) measured by `run.jl` |
| `ci/` | the pull-request benchmark: a small BenchmarkTools suite and the driver that posts its base/head comparison |
| `large_datasets/` | Ristretto alone on two full-size scanner datasets ([its README](large_datasets/README.md)) |
| `slurm/` | SLURM scripts; machine paths live in the untracked `slurm/site.env` |
| `results/` | harness results and SLURM logs (gitignored) |

`benchmark/` is a member of the root workspace (`Project.toml`), so it runs against this checkout of
Ristretto.

## The case catalog

`utils/bench_utils.jl` defines `module BenchUtils`. `get_case(id)` returns a `BenchCase`: the
zero-filled k-space, mask or trajectory, sensitivity maps and reference image of one problem,
memoised per process. Synthetic data come only from GeometricMedicalPhantoms, FFTW and explicitly
seeded RNGs. Ristretto is used only to simulate non-Cartesian data (NFFT at `m = 8, σ = 2`), and to run
ESPIRiT on real data. Ristretto's own simulation code can therefore change between two measured refs
without changing either one's input.

| id | problem | sampling |
|---|---|---|
| `shepp_logan_2d_1ch_cartesian` | 128², 1 coil | variable-density random ky, 16 ACS lines, R ≈ 2.5 |
| `shepp_logan_2d_8ch_cartesian` | 128², 8 coils | variable-density ky + 16 ACS, R = 4 |
| `shepp_logan_2d_8ch_radial` | 128², 8 coils | golden-angle radial, 256 samples × 64 spokes, ramp DCF |
| `shepp_logan_multislice_8ch_cartesian` | 12 slices of the 128³ volume, maps from a 3D field | one ky mask for every slice, R = 4 |
| `shepp_logan_3d_8ch_cartesian` | 128³, 2 rings × 4 coils of true 3D maps | variable-density ky–kz, 24² calibration, R = 6 |
| `torso_cine_8ch_cartesian` | torso cine, 128², one cardiac cycle | per-frame random ky, 8 centre + 24 random lines |
| `torso_cine_8ch_radial` | the same series | golden-angle radial 256 × 34 per frame (one trajectory shared by every frame) |

All are `ComplexF32`, noise at `RISTRETTO_BENCH_SNR_DB` (30 dB) on unit-RMS k-space, with a fixed seed
per id. The 3D volume and the cines are **heavy**: one warm-up and one timed run, where every other
case takes the minimum of three runs.

`applicable_methods(case)` lists what is run on a case: the adjoint (or the DCF-weighted gridding
for radial data) and CG-SENSE for every case, then TV (isotropic `tv` and anisotropic `atv`),
L1-wavelet and TGV for the static Cartesian ones (TV and wavelet for the volume, TV for radial),
and global low rank, locally low
rank and temporal TV for the cines. Every TV also runs by PDHG instead of ADMM (`tv_pd`, `atv_pd`,
`ttv_pd`) at `CMP_OUTER × CMP_CG_ITERS` iterations; a checkout without `ChambollePock` skips those
rows. Iteration counts, λ and the ADMM ρ are fixed
(`utils/ristretto_methods.jl`, `CMP_OUTER` / `CMP_CG_ITERS` to override).

`get_case(id; pattern = :regular)` is a regularly undersampled variant of the 2D and multislice
cases, for GRAPPA, which needs a fixed stride.

`RISTRETTO_BENCH_SMALL=1` shrinks every case: 32², 3 slices, 32³, 8 frames and 4 coils. It is meant for
smoke runs on a login node and for the `Benchmark case catalog smoke test` TestItem.

`utils/check_cases.jl` checks the catalog:
- the array layouts agree;
- the acceleration is within 5% of its design;
- no k-space sample outside the mask is nonzero;
- a noiseless, fully sampled Cartesian adjoint reproduces the phantom (NRMSE < 1e-5);
- a noiseless, Nyquist-sampled radial gridding with a Pipe-Menon DCF reproduces the phantom band-limited to the |k| ≤ ½ disc a radial trajectory covers (NRMSE < 0.15; a transposed or shifted convention scores above 1). The band limit is needed whatever the DCF: the corners outside the disc hold 14% of Shepp-Logan's energy and are never sampled.

It also prints a hash of every k-space, for comparing processes:

```sh
RISTRETTO_BENCH_SMALL=1 julia --project=benchmark benchmark/utils/check_cases.jl [--cases=radial] [--real]
```

### Real data

Every synthetic case has a real analogue (`real_*` ids, `REAL_CASES`), assembled by
`utils/real_data.jl` from [MRITestData.jl](https://github.com/hakkelt/MRITestData.jl). For each one
there is an open dataset and, where it is better suited, a gated fastMRI one. The open dataset is
preferred when it is as good; a real case falls back to the other when its first choice is
unavailable, and records which one it used (`data_source`).

| real id | first choice | fallback |
|---|---|---|
| `real_2d_1ch_cartesian` | fastMRI `singlecoil_val/file1000107` (native single channel, gated) | M4RAW, SENSE-combined to one channel |
| `real_2d_multichannel_cartesian` | M4RAW `multicoil_train/2022062402_T203`, middle slice | — |
| `real_2d_multichannel_radial` | fastMRI breast `fastMRI_breast_006_2`, central partition, 128 golden-angle spokes (gated) | USC speech spiral, one frame |
| `real_multislice_multichannel_cartesian` | M4RAW, 12 slices | fastMRI brain `file_brain_AXFLAIR_203_6000923` |
| `real_3d_multichannel_cartesian` | MRIDATA knee `52c2fd53-…`, a true 3D encode cropped to 128³ | — |
| `real_cine_multichannel_cartesian` | OCMR `fs_0001_1_5T` | — |
| `real_cine_multichannel_radial` | USC speech spiral real-time (3 of 13 arms per frame) | — |

Cartesian real data are fully sampled scans undersampled retrospectively with the analogue's
pattern. The reference is the SENSE combination of the full k-space, so `nrmse_gt` measures accuracy
here too. A real case uses the λ calibrated for its synthetic analogue. That λ transfers because
both k-spaces are unit-RMS normalised.

A gated source counts as available when its files are cached, or when signed fastMRI URLs are
registered in the environment's gitignored `LocalPreferences.toml`. `RISTRETTO_BENCH_REAL_PREFER=open` or
`gated` overrides the order. Prepared cases (ESPIRiT maps included, 3–4 minutes for the 128³ knee)
are cached in `RISTRETTO_BENCH_WORK_DIR`.

## The Ristretto harness: `run.jl`

```sh
julia --project=benchmark -t 8 benchmark/run.jl --threads=8 [--use-mkl] [--cases=shepp_logan_2d] [--methods=tv,cgsense] [--real]
```

Each (case, method) pair is timed with `BenchUtils.time_run`: one warm-up, then the minimum and
median of three runs, or of one for a heavy case. NRMSE is taken against the case's reference. Each
case writes one immutable JSON file to `results/runs/`. It records:
- threads, backend, host, pinned CPUs and node class (CPU model + SLURM partition);
- the git ref;
- tree hashes of Ristretto's `src/`, `ext/`, `Project.toml` and `deps/`, plus the NestedThreading checkout;
- the environment variables that change a measurement (`KMP_BLOCKTIME`, `OPENBLAS_THREAD_TIMEOUT`, …).

**Measure once.** A configuration already stored for the same *clean* code on the same node class
is skipped ("stored, skipping"). A baseline is therefore measured once, and later iterations of a
branch only pay for the branch. The code key hashes the package code, not the commit, so a commit
that only touches `benchmark/` keeps the stored results. `--remeasure` forces a rerun. A dirty
checkout is never stored as reusable.

**Another checkout.** `--ristretto=PATH` measures the Ristretto checkout at `PATH`, a worktree of master for
example. The same harness re-runs itself in `PATH/benchmark`'s environment. `--dev=Pkg=PATH` points
that environment's manifest entry of `Pkg` at another checkout
(`--dev=NestedThreading=/path/to/checkout`). `--ref-name=NAME` labels the results (default: the
branch name).

```sh
git worktree add ../ristretto-master master
julia --project=benchmark -t 8 benchmark/run.jl --threads=8 --ristretto=../ristretto-master --ref-name=master
julia --project=benchmark -t 8 benchmark/run.jl --threads=8 --ref-name=perf
julia --project=benchmark benchmark/compare.jl master perf [--threads=8] [--backend=openblas]
```

`compare.jl A B` matches the latest result of each side by (case, method, threads, backend,
environment variant). It prints the B/A ratio of the minimum time, each side's spread (median /
minimum) and ΔNRMSE. |ΔNRMSE| > 1e-3 is flagged as a change in *what* is computed, and a ratio outside
the larger spread plus 5% as a real speed change. Results from different node classes are warned
about, not compared silently. Only isolated runs count unless `--placement=shared` or `any` is given:
a packed run (`matrix.sh --pack`) recorded after an isolated one would otherwise be the one compared.
`--pick=min` takes the fastest matching result instead of the latest, for repeated measurements
(`matrix.sh --swap-repeat`).

## Pull-request benchmarks: `ci/`

`.github/workflows/benchmark.yml` runs on every pull request to `master` and posts one comment,
updated on each push, comparing the pull request with its base. `ci/benchmarks.jl` is a
BenchmarkTools `SUITE` of small synthetic catalog cases (`RISTRETTO_BENCH_SMALL` sizes):

- `solve/<case>/<method>`: `reconstruct` at a fixed iteration count, with the magnitude NRMSE
  against the ground truth beside the time;
- `operator/<case>/{forward,adjoint,normal}`: `mul!` with the encoding operator;
- `setup/<case>/{operator,opnorm}`: building the encoding operator and estimating its norm.

`ci/compare.jl` runs the head's suite against both revisions, each in its own copy of the `ci/`
environment with that revision developed into it, once per thread count (`-t 1`, `-t 2`). Two
processes of the same code differ by more than the spread within one: comparing master with itself
on the login node (2026-10-09) flagged 2–4 of 35 entries per thread count at up to 1.7×, with one
process per side or with the best of two. `--repeats n` runs each side in `n` alternating processes
and keeps each entry's fastest; it did not remove those flags and doubles the run (each process
spends about 3 minutes compiling for 25 seconds of timing), so CI runs one. An entry
that fails on one revision (the base may lack what the head's suite uses) shows `—` and is listed.
Run it locally against another checkout with

```sh
git worktree add --detach /tmp/base master
julia --project=benchmark/ci -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/ci benchmark/ci/compare.jl --base-dir /tmp/base --head-dir . --output-dir /tmp/bench-out
```

The ratio is base/head, so above 1 the pull request is faster. 🚀 and 🐢 mark a time ratio beyond 1.2
or below 0.8 outside the interquartile error, or a memory ratio beyond 1.5 or below 0.5; ⚠️ marks a
solve whose NRMSE moved by more than 5 %. Shared runners are noisy: these are tripwires, not
measurements. Timing claims come from `run.jl` on a cluster node.

## SLURM

Copy `slurm/site.env.example` to `slurm/site.env` and fill in the paths of this machine: the Julia
depot, the data directory and the partition names. No tracked
file names a machine path. Every script sources `slurm/common.sh`, which sets:
- the depot Julia binary (the login-node `julia` wrapper fails inside a SLURM cgroup);
- `TMPDIR=/dev/shm`, for BART's `.cfl` files;
- `KMP_BLOCKTIME=0`.

Submit through `slurm/submit.sh`, from the repository root. It uses the **test** partition (one hour)
unless given `--production`, which occupies a node for hours and needs agreement first.

| script | what it runs |
|---|---|
| `matrix.sh` | one exclusive node; each configuration pinned to its own NUMA domain with `numactl`, several at a time, one CPU per physical core (never an SMT sibling; `run.jl` pins its threads only to physical cores of the allowed set, and fails when it has too few). Refs and environment variants of one (threads, backend) pair start together. It runs `run.jl`. Matrix dimensions: `--matrix-threads=`, `--matrix-backends=`, `--matrix-refs=name:path[:Pkg=PATH+...],...` (the optional third field is passed to that ref's `run.jl --dev`), `--matrix-env=A=1+B=2,...` (recorded per row). The first ref or variant of a pair always starts on the lower domain, and domains differ; `--swap-repeat` queues the matrix again in reverse order with `--remeasure`, to be compared with `compare.jl --pick=min`. `--pack` lets several tasks share a domain, each on its own physical cores (best-fit within an L3 group, then a domain) and with `--pack-mem-gb=` reserved per task against the domain's memory; packed results get their own node class, so they are never reused for or silently compared with isolated ones. Leave it off for published numbers and when memory-bandwidth contention is the question. Every other flag is passed through. |
| `large_datasets.sh`, `large_datasets_gpu.sh` | `large_datasets/run.jl` on the host and on one GPU |
| `gpu_refs.sh` | a GPU benchmark on several checkouts in one job, alternating them round by round |
| `report_efficiency.sh <jobid>` | CPU efficiency of a finished matrix job |

```sh
benchmark/slurm/submit.sh matrix.sh --matrix-refs=master:../ristretto-master,perf:. --matrix-threads=1,8
```

Logs go to `results/slurm/`.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `RISTRETTO_BENCH_SMALL` | `0` | shrink every case (smoke runs) |
| `RISTRETTO_BENCH_REAL_DATA` | `0` | include the real-data analogues in `run.jl` (same as `--real`) |
| `RISTRETTO_BENCH_SNR_DB` | `30` | noise level of the synthetic cases |
| `RISTRETTO_BENCH_CINE_FRAMES` | `30` | frames of the cine cases |
| `RISTRETTO_BENCH_DATA_DIR` | MRITestData's cache | where real data are downloaded |
| `RISTRETTO_BENCH_WORK_DIR` | `<data dir>/ristretto_benchmark` | prepared real-data cases |
| `RISTRETTO_BENCH_REAL_PREFER` | — | `open` or `gated`: override the dataset order |
| `CMP_OUTER` / `CMP_CG_ITERS` | `20` / `10` | outer iterations / inner CG iterations and CG-SENSE iterations |

Each provider's own licence and citation terms apply to the real data; see
<https://hakkelt.github.io/MRITestData.jl/stable/legal/>.

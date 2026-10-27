# Performance & Threading

Reconstruction speed on a multi-core machine depends on a handful of settings that are easy to
get wrong. This page covers what you need to do; Ristretto handles the rest.

## Quick checklist

```bash
export KMP_BLOCKTIME=0          # only if you use MKL
julia --project -t 8 recon.jl   # -t = number of cores you actually have
```

- Give Julia as many threads as you have physical cores — no more.
- On Slurm, make `--cpus-per-task` match `-t`, and always ask for memory explicitly.
- Don't call `BLAS.set_num_threads` yourself. Ristretto manages it during reconstruction.

That is usually all you need.

## Why threading can make things slower

Julia, BLAS, FFTW and NFFT each have their own thread pool, and by default each one sizes itself
from the machine without knowing about the others. Nest them and you get more threads than cores,
all paying synchronisation costs on work that was already busy.

MRI reconstruction makes this worse than usual, because the natural unit of work is small — a
128×128 slice is 1 MiB and fits in cache. Splitting *that* across 8 cores costs more in
coordination than it saves in arithmetic.

The right structure is: **parallelise across slices, keep each slice serial.** Ristretto does this for
you. Problems appear when something underneath the slice loop starts its own threads anyway.

Concretely, on an 8-core run of a typical iterative solve:

| BLAS threads | time    |
|--------------|---------|
| 1            | 2.7 s   |
| 8            | 15.6 s  |

Same code, same cores — nearly 6x slower just from letting BLAS thread. OpenBLAS and MKL behave
the same way here; this is not a bug in either.

## What Ristretto does for you

- **Splits work across slices** and tells the libraries underneath to stay single-threaded.
- **Skips threading for small FFTs**, where it would cost more than it saves.
- **Leaves the per-kernel threading decision to the operator.** Whether a given kernel is worth
  threading depends on that kernel and the size it is actually handed, so it is decided by
  `AbstractOperators.threading_threshold` and `ProximalOperators.should_thread`, per operator,
  per input — not by one rule applied to a whole solve.
- **Spreads slices of a task-split problem over threads when there are enough of them**, and then
  runs the work *inside* a slice sequentially, because the slice loop is already using every
  thread. With a `SequentialExecutor` the threads are free and the inside threads as usual.
- **Chooses how carefully to plan FFTs**, from the size of the problem, the algorithm and the
  iteration count (see [Planning FFTs](@ref) below).
- **Remembers measured FFT plans across sessions.** What FFTW learns while measuring a plan (its
  "wisdom") is saved to a file per machine, and later sessions load it before planning, so even
  an `ESTIMATE` plan of a problem measured before comes out measured (see below).

## Planning FFTs

Before its first transform, FFTW makes a *plan*. With `ESTIMATE` it guesses one from a heuristic
in microseconds; with `MEASURE` it times candidate algorithms on the actual arrays, which costs
about 0.1–0.2 s per 2D transform and 1–1.5 s per 3D one, whatever the number of coils or frames.
An operator plans about four transforms (forward, adjoint, and the normal operator's pair), so a
cold `MEASURE` build costs 0.5–0.8 s for a 2D problem.

What `MEASURE` buys back depends mostly on the grid. Single transforms, one thread, planning cost
/ run time per batch of 8 coils:

| grid | `ESTIMATE` | `MEASURE` | `ESTIMATE` is slower by |
|------|-----------:|----------:|------------------------:|
| 128² | 0 ms / 0.46 ms | 123 ms / 0.25 ms | 1.8× |
| 256² | 1 ms / 13.6 ms | 185 ms / 1.5 ms  | 9× |
| 128³ | 0 ms / 161 ms  | 910 ms / 62 ms   | 2.6× |

A 256² grid is what a 128² non-Cartesian image is interpolated to, so radial and spiral
reconstructions are where `ESTIMATE` hurts most: on one thread, a radial cine low-rank solve took
158 s with `ESTIMATE` plans and 25 s with `MEASURE` ones. More FFTW threads shrink the gap: at 16 threads the
128² transform runs equally fast either way.

### The automatic choice

`fft_planning = :auto` (the default) adds up a score, each term the `log2` of one factor of
"does the time saved pay for the planning":

- **the transforms the reconstruction will run**: iterations × operator applications per
  iteration × 2 (forward and adjoint). An ADMM iteration counts one application per inner CG
  step plus one; CG, FISTA, POGM and PDHG count one. A direct reconstruction counts a single
  transform;
- **the batch**: coils × frames × every other axis transformed together;
- **the grid**: a larger grid gains more from a measured plan per unit of planning time, steeply
  so between 128² and 256²;
- **the threads**: every doubling of FFTW's threads counts against `MEASURE`;
- **a constant** for the four transforms each operator plans.

`MEASURE` is used when the sum is not negative. Checked against 36 whole reconstructions
planned from scratch (2D Cartesian and radial, 3D, Cartesian and radial cine; CG-SENSE, ADMM
and PDHG; 1, 4 and 16 threads), it picked the faster planner in every one that was not a tie:

| case | `ESTIMATE` | `MEASURE` | chosen |
|------|-----------:|----------:|--------|
| 2D 128², 8 coils, CG-SENSE 10 it, 1 thread | 0.014 s | 0.24 s | `ESTIMATE` |
| 2D radial, 8 coils, CG-SENSE 10 it, 1 thread | 0.30 s | 0.84 s | `ESTIMATE` |
| 2D radial, 8 coils, TV (ADMM) 20 it, 1 thread | 5.46 s | 1.64 s | `MEASURE` |
| 2D radial, 8 coils, TV (ADMM) 20 it, 16 threads | 0.86 s | 1.00 s | `ESTIMATE` |
| 3D 128³, 8 coils, TV (ADMM) 20 it, 4 threads | 31.3 s | 22.5 s | `MEASURE` |
| cine 128², 8 coils, 30 frames, low-rank 20 it, 16 threads | 2.74 s | 2.95 s | `ESTIMATE` |
| radial cine, 8 coils, 30 frames, low-rank 20 it, 16 threads | 14.3 s | 4.96 s | `MEASURE` |

The timings are from one node; FFTW's heuristic differs between CPUs, so the margins will too.

### Choosing yourself: `fft_planning`

The automatic choice looks at one reconstruction in a fresh process. It cannot know that you are
about to run the same acquisition again — a λ sweep, several regularizers, a direct
reconstruction for a preview followed by an iterative one. Once any plan of a transform has been
measured, every later plan of the same transform in the session reuses it, including an
`ESTIMATE` one, and it is saved to the wisdom cache for later sessions. So for repeated work,
measure once:

```julia
config = ReconstructionConfig(; fft_planning = :measure)
x̂₀ = reconstruct(acq, DirectReconstruction(); config)   # pays for the measured plans
x̂s = map((1e-4, 3e-4, 1e-3)) do λ                         # reuse them: plan in microseconds
    reconstruct(acq, IterativeReconstruction(TotalVariation2D(λ)); config)
end
```

Even a single direct reconstruction is worth measuring this way when it will be repeated: on a
256² grid, a measured plan saves about 12 ms per transform of 8 coils, so it has paid for itself
after about 15 transforms.

`fft_planning = :estimate` does the opposite: it never measures, for a one-off script where
start-up time matters more than the solve. `fft_planning` has no effect under FFTW.jl's `mkl`
provider: MKL's FFTW interface ignores planner flags, keeps no wisdom, and plans quickly anyway.

Planning the same transform with `MEASURE` again does not find a better plan: FFTW times the same
candidates each time, and five fresh `MEASURE` plans of each grid above ran equally fast. A
more thorough search is `PATIENT`, below.

### The FFT wisdom cache

The cache lives in Ristretto's scratch space (`~/.julia/scratchspaces/<uuid>/fftw_wisdom/`), or under
`$XDG_CACHE_HOME` (else `~/.cache`) when that is not writable. A binary built with JuliaC (without
`--trim`; FFTW.jl does not pass trimming) resolves the same scratch space under `~/.julia` with no
Julia installed, so its first run on a machine creates the cache. The file is
named after the CPU model and the FFTW version, since wisdom from one CPU does not carry over to
another, and machines sharing a home directory keep separate files.

Wisdom holds for one exact transform — grid size, coil and frame count, and the number of FFTW
threads. Entries for different thread counts sit side by side in the file, but each session only
benefits from the ones made with its own. That also rules out making wisdom while a package
precompiles: precompilation runs on one thread, and timing a multithreaded plan there picks one
for serial execution.

To plan a recurring problem more carefully than Ristretto does on its own, run
[`plan_fft_wisdom`](@ref) once per machine, with the thread count reconstructions will use and on
an acquisition shaped like theirs:

```julia
# julia -t 16
using Ristretto
plan_fft_wisdom(acq; rigor = :patient)   # 10–35 s per 2D transform, minutes per 3D one
```

`PATIENT` searches more candidates than `MEASURE`, and the gain depends on the transform: at 4
threads it found plans 10–20 % faster for 256²×8 (0.78 against 0.99 ms per transform) and 128³×8
(35 against 40 ms), while on another node it was 15–20 % faster for 256²×8 and no faster for
128²×8 or 128²×8×30. It pays only for long solves and for problems reconstructed many times.
Capping its planning time (FFTW's `timelimit`) does not help: with 0.1 s it falls back to plans
as slow as `ESTIMATE`'s, with 0.5 s it matches `MEASURE` in 2D but not in 3D, and with 2 s it
matches `MEASURE` at several times the planning cost. The environment variable
`RISTRETTO_FFTW_WISDOM` turns the cache off (`off`) or moves it (a directory); the benchmark harness
turns it off, so that every timing plans from scratch.

Earlier versions narrowed every pool for the duration of any sub-16-MiB solve. That is gone: it
was measured against a `Polyester`/`Threads.@threads` interference cost that `NestedThreading`
has since removed, and it silently serialised every `Polyester.@batch` kernel as a side effect.
Re-measured on a 128²×8-coil phantom at 8 threads (min of 15), opening up is worth 1.45x on an
L1-Wavelet FISTA solve and costs 1.07x on a TV-ADMM one.

## What you have to set yourself

### `KMP_BLOCKTIME=0` — MKL only, before Julia starts

Intel's MKL keeps its worker threads spinning at full CPU for 200 ms after finishing each piece
of work. Those spinning threads crowd out the Julia threads doing the FFTs and prox steps.

```bash
export KMP_BLOCKTIME=0
```

On the benchmark above, with MKL and 8 BLAS threads, this is the difference between 16.1 s and
3.8 s.

!!! warning "This cannot be set from inside Julia"
    MKL reads the variable once, when it loads, which happens before any of your code runs.
    Setting `ENV["KMP_BLOCKTIME"]` in a script or a package is too late to have any effect, and
    the runtime function that claims to change it (`kmp_set_blocktime`) succeeds while silently
    doing nothing. It has to come from the shell, a job script, or your `.bashrc`.

    Ristretto warns you if it sees MKL loaded without it.

### Slurm

```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
julia --project -t 8 recon.jl
```

Ask for memory explicitly — the default per-CPU allowance is small, and a single-core job will be
killed by a reconstruction that would otherwise fit in a few GB.

Julia detects your allocation correctly, so BLAS sizes itself to `--cpus-per-task`. Note that
`Sys.CPU_THREADS` reports the whole node regardless of what you were given; don't use it to size
anything.

### Pinning threads to cores

```julia
using ThreadPinning
pinthreads(:affinitymask)   # stays within your Slurm allocation
```

This matters when the OS scheduler would otherwise move your threads around and cost you memory
locality:

- **Multi-socket / multi-NUMA machines** — most HPC compute nodes, dual-socket workstations, and
  the larger Threadripper/EPYC desktops. A thread that migrates to the other socket reads its
  data across the interconnect. This is the big one.
- **Shared machines** — a login node or a workstation with other users, where contention makes
  the scheduler reshuffle runnable threads.
- **Benchmarking** — pinning removes a large source of run-to-run variance regardless of topology.

On a single-socket machine that you have to yourself (a typical laptop or desktop) it does
little, and aggressive pinning can fight the OS power management. `pinthreads(:affinitymask)` is
the safe default: it pins within whatever cores you were actually given and is a no-op if that
is the whole machine.

### OpenBLAS or MKL?

Either is fine. They perform the same on typical reconstruction work once threading is set up
correctly. MKL is somewhat better at the SVDs in low-rank methods; its downside is the
`KMP_BLOCKTIME` trap above. Loading `MKL.jl` replaces OpenBLAS rather than running alongside it,
so there is nothing to balance between them.

## If a reconstruction is slow

Try this first:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```

If that makes it faster, some library underneath the slice loop is threading when it shouldn't
be. Report it — that is a bug in Ristretto's threading setup, not something you should have to work
around.

Otherwise, check what is actually configured:

```julia
using LinearAlgebra, FFTW
@show Threads.nthreads() BLAS.get_num_threads() FFTW.get_num_threads()
@show get(ENV, "KMP_BLOCKTIME", "unset")
```

## Non-Cartesian accuracy / speed trade-off

`get_fourier_operator`/`get_encoding_operator` take `m`, `sigma` and `precompute` keywords that
forward straight to `NFFTOp`/NFFT.jl, exposing the gridding operating point instead of leaving
it fixed. Left at `nothing` (the default), **Ristretto's own default operating point is used**: `m = 3`,
`σ = 1.5`, `precompute = NFFT.TENSOR` (`DEFAULT_NFFT_M`, `DEFAULT_NFFT_SIGMA`,
`DEFAULT_NFFT_PRECOMPUTE` in `src/encoding/fourier_operators.jl`).

The default was NFFT.jl's own (`m = 5`, `σ = 2.0`, `NFFT.POLYNOMIAL`) at first, then `m = 4`,
`σ = 1.5`; both are far more accurate than the reconstruction needs. Measured on a 128×128 radial phantom
(`GeometricMedicalPhantoms`'s Shepp-Logan, 256 samples × 128 spokes), single thread, timings
interleaved round-robin across configs rather than one after another (a single measurement on
this shared login node can swing 30-60%):

| m | σ | precompute | forward (min/median ms) | adjoint (min/median ms) | forward rel. error vs `m=5,σ=2` |
|---|---|---|---|---|---|
| 5 | 2.00 | POLYNOMIAL (first default = NFFT.jl's own) | 8.5 / 9.7 | 7.5 / 8.6 | 0 (reference) |
| 4 | 2.00 | POLYNOMIAL | 7.0 / 8.1 | 5.5 / 6.4 | 3.8e-8 |
| 4 | 1.50 | POLYNOMIAL (former default) | 4.0 / 4.5 | 4.6 / 5.3 | 2.5e-7 |
| 3 | 2.00 | POLYNOMIAL | 6.0 / 6.8 | 4.3 / 4.9 | 2.4e-6 |
| 3 | 1.50 | POLYNOMIAL | 2.8 / 3.2 | 3.3 / 3.8 | 1.7e-5 |
| 3 | 1.25 | TENSOR (MRIReco's point) | 2.5 / 2.9 | 2.8 / 3.1 | 7.1e-5 |
| 2 | 1.50 | POLYNOMIAL | 2.3 / 2.6 | 2.4 / 2.8 | 7.4e-4 |
| 2 | 1.25 | TENSOR | 2.0 / 2.3 | 1.9 / 2.2 | 2.1e-3 |

The direct (gridding) reconstruction's NRMSE against the phantom does not move outside
run-to-run noise across this whole table (consistent with the older measurement below, where
NRMSE was ≈ 0.085 at both the old default and MRIReco's point) — non-Cartesian gridding-adjoint
NRMSE is dominated by sampling/DCF artifacts, not by the gridding kernel's own accuracy, so
forward relative error against the reference is the right proxy for "is this operating point
accurate enough."

Iterative reconstructions are just as insensitive. Each method of the `benchmark/` harness was
run at every operating point on the radial Shepp-Logan case (8 coils, 64 spokes) and the radial
torso cine (8 coils, 34 spokes × 30 frames), in `ComplexF32`, and compared with the same
reconstruction at `m = 8, σ = 2` (NFFT error measured against that transform as well):

| NFFT error | operating point | change in magnitude NRMSE vs the `m=8` reconstruction |
|---|---|---|
| 6e-6 | `m=4, σ=1.5` | ≤ 3e-5 |
| 5e-5 | `m=3, σ=1.5` (default) | ≤ 4e-5 (CG-SENSE, L1-wavelet, TV ADMM / Chambolle-Pock, low rank, temporal TV) |
| 4e-4 – 7e-4 | `m=3, σ=1.25`; `m=2, σ=2` | ≤ 3e-4 |
| 2e-3 | `m=2, σ=1.5` | up to 9e-4 (temporal TV 1.4% worse, TV ADMM 1% worse) |
| 7e-3 | `m=2, σ=1.25` | up to 3e-3 (TV ADMM 9% worse) |

The NRMSEs themselves are 0.03-0.27, so the default's change is at most 0.13% of them; accuracy
starts to matter near 1e-3 NFFT error. `m = 3, σ = 1.5` is the default because it is the fastest
point clearly below that: 1.5-1.6x faster per forward/adjoint pair than `m = 4, σ = 1.5`
(8 threads, EPYC 7763: 2D 256² radial, 12 coils, 52 vs 76 ms; 3D 128³ kooshball, 4 coils,
0.77 vs 1.24 s).

Independently, MRIReco's operating point (`m = 3`, `σ = 1.25`, `NFFT.TENSOR`) is faster still, at
real accuracy cost: measured per coil, 4.28 ms vs 1.00 ms for a forward error of 1.6e-7 vs 5.7e-5
against Ristretto's old (`m=5,σ=2`) default, while the reconstructed image's NRMSE is 0.085 either way
(measured whole multi-coil DCF adjoint, single thread, `benchmark/comparison/scripts/
run_noncart.jl`, 2026-09-04: Ristretto at its old default 25.9 ms, Ristretto at MRIReco's operating point
6.1 ms, MRIReco 27.3 ms — all at NRMSE ≈ 0.085 against the phantom). At MRIReco's operating point
Ristretto is faster than MRIReco while gridding more accurately; ask for it explicitly if you want the
faster, less accurate end of the curve:

```julia
𝒜 = get_encoding_operator(info; m = 3, sigma = 1.25, precompute = NFFT.TENSOR)
```

or go back to the old high-accuracy default with `m = 5, sigma = 2.0`.

## Notes for developers

- `serial_blas_threshold_bytes()` (16 MiB by default) is the size above which Ristretto stops forcing
  serial BLAS — and, since it also gates whether a solve threads at all, the size above which a
  small problem is allowed to use more than one thread. Re-fitted on real solves on an exclusive
  node, the crossover depends on the BLAS backend: below 4 MiB the threaded and serial paths are
  within 2% of each other on both, but at 8 MiB threading is a 1.3x *loss* on OpenBLAS and a
  1.4x *win* on MKL. 16 MiB is the value that is safe on both. Move it with
  `Ristretto.set_serial_blas_threshold_bytes!` or the
  `RISTRETTO_SERIAL_BLAS_THRESHOLD_BYTES` environment variable if you know your backend and hardware.
- `NestedThreading`'s `exclude` keyword honours counted pools since 0.1.1, and `only` gives the
  allowlist form. Before that release `exclude` affected Polyester alone and naming
  `:blas`/`:mkl`/`:fftw`/`:nfft` did nothing silently, which is why the BLAS budget used to be
  narrowed from the inside instead.
- Every iterative solve runs inside `with_serial_blas`, which is
  `with_thread_default(f, 1; only = (:blas, :mkl))`: serial BLAS *by default*, since almost every
  BLAS call in a solve is a level-1 update too small to thread. The calls that are large enough
  take BLAS's threads back with `NestedThreading.with_thread_grant`, each behind its own size gate:

  | call | knob (a `Ref`) | default |
  |------|----------------|---------|
  | dense `svd!`/`eigen!` in a low-rank prox | `ProximalOperators.FACTORIZATION_THREAD_WORK` | `m·n·min(m,n)` ≥ 2^22 |
  | `MatrixOp`/`LMatrixOp` `gemm` | `AbstractOperators.BLAS3_THREAD_WORK` | `m·n·k` ≥ 2^25 |
  | a CG step's `dot`/`axpy!` | `ProximalAlgorithms.CG_BLAS_THREAD_BYTES` | 8 MiB on MKL, never on OpenBLAS |

  A grant never goes past a hard limit, so inside a slice loop that already occupies every
  thread nothing changes. Set a knob to `typemax(Int)` to keep that kind of call serial. On
  OpenBLAS, closing a grant also shuts the BLAS worker threads down
  (`NestedThreading.park_openblas`), which would otherwise spin for about 0.1 s on the cores the
  next FFT needs; this costs about 1 ms per grant, which is what the gates are sized against.
  A CG step is too short for either: granting OpenBLAS at every step of a 128³ solve made it
  1.5x slower, which is why CG never grants OpenBLAS by default.
- Restricting BLAS is not the same as restricting the process: `with_restricted_threads` also
  narrows FFTW and NFFT and switches Polyester off, so `BLAS.get_num_threads() == 1` is never
  evidence that entering the scope would be a no-op.
- `with_full_threads` raises thread counts to capacity, overriding a lower count you set
  deliberately. Don't wrap one around hand-tuned settings.
- `--gcthreads` does not help the residual `-t 8` cost (GC growing from ~10 ms to ~35-46 ms per
  solve on an allocation-heavy serial solve): measured at `--gcthreads=1`, no change. The lever
  for that residual is allocation, not GC thread count — see the per-iteration prox-buffer work
  tracked as `IMPLEMENTATION_PLAN.md`'s `C7`.

## API

```@docs
plan_fft_wisdom
Ristretto.fftw_wisdom_path
Ristretto.with_serial_blas
Ristretto.serial_blas_threshold_bytes
Ristretto.set_serial_blas_threshold_bytes!
Ristretto.DEFAULT_SERIAL_BLAS_THRESHOLD_BYTES
```

# benchmark/startup/

What a user pays before the first image, which the cross-toolkit tables in
`docs/src/related_packages.md` leave out: those report warm solves (the fastest of three after a
warm-up, see [`../comparison/README.md`](../comparison/README.md)), while a BART call is a fresh
process every time, a Python toolkit has to be imported, and a Julia toolkit also compiles its solve
on the first call.

`run.jl` measures, in fresh processes, on `shepp_logan_2d_8ch_cartesian` (128², 8 coils) for the
L1-wavelet row (FISTA, 20 iterations; not MIRT, which has no wavelet prox) and the CG-SENSE row
(10 iterations; every toolkit):

| quantity | what is timed |
|---|---|
| `runtime_s` | a process that does nothing: `julia -e 0`, `python -c pass` |
| `import_s` | a process that only loads the toolkit: `using Ristretto` / `MRIReco` / `MIRT`, `import sigpy, sigpy.mri.app`, `import mrpro`; `bart version` |
| `first_solve_s` | the first solve in a fresh process, compilation included |
| `warm_solve_s` | the median of the 10 solves after it in the same process |
| `end_to_end_s` | a fresh process that loads the toolkit, reads the case and solves it once |
| `ristretto_precompile_s` | once: `Base.compilecache` of Ristretto into an empty depot, its dependencies' caches already present |

Every Julia number except the last assumes the packages are already precompiled. For BART every
solve is its own `bart pics` process on `.cfl` files in `/dev/shm`, so first, warm and end-to-end
are one number. The comparison tables subtract BART's process spawn and file I/O from that number
(`bart_overhead` in `../comparison/scripts/_setup.jl`); this one keeps them.

Each toolkit's solve is a standalone script (`solve_<toolkit>.jl` / `.py`) that loads only that
toolkit and repeats the call the comparison harness times (`../comparison/scripts/_toolkits.jl`,
`../utils/ristretto_methods.jl`) with the same parameters and calibrated λ; the NRMSE each reports
matches the harness's row. `prepare.jl` writes the case once, from the case catalog, as raw arrays
and BART `.cfl` files.

Every child runs pinned to the same 8 physical cores of one NUMA domain (`numactl`), with
OpenMP / OpenBLAS / MKL / PyTorch at 8 threads, Julia at `-t 8`, and BLAS and FFTW as
`../comparison/scripts/_setup.jl` sets them (MRIReco keeps its own BLAS choice). BART is the
OpenBLAS build. Toolkits are interleaved within each repetition.

## Running it

On a compute node, through SLURM (one hour on the test partition is enough):

```sh
benchmark/slurm/submit.sh benchmark/startup/startup.sh
```

or directly, with the machine paths in `benchmark/slurm/site.env`:

```sh
julia --project=benchmark/comparison benchmark/startup/run.jl
```

`STARTUP_THREADS` (8), `STARTUP_REPS` (5 fresh processes per load and end-to-end measurement),
`STARTUP_WARM_REPS` (3 fresh processes timing first and warm solves), `STARTUP_NWARM` (10),
`STARTUP_PRECOMPILE` (1), `STARTUP_DATADIR` and `STARTUP_OUT` (`results.json` here) override the
defaults. `results.json` holds the per-toolkit medians (`summary`), every sample (`raw`) and the
node, CPU, versions and pinning.

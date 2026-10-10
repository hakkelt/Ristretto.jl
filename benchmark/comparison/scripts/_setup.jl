# Shared prelude for the decomposed comparison suite. Every `run_<section>.jl` does
# `include(joinpath(@__DIR__, "_setup.jl"))` first: it parses the CLI, pins threads, configures
# the BART / OpenMP / MKL environment, loads Ristretto + BART + SigPy + MRIReco and the case catalog
# (benchmark/utils/), and defines the timing helpers and `BenchResult`. Sections take their data
# from the catalog only.
#
# Each section script then appends to `results::Vector{BenchResult}` and calls
# `write_section("<name>")`, which records one immutable run file under
# `results/runs/` via `ResultsStore.jl`. `query_results.jl` reads that directory back for
# analysis -- there is no merge step (see `ResultsStore.jl`'s module docstring for why).

using Printf
using JSON

const USE_MKL = "--use-mkl" in ARGS
let i = findfirst(a -> startswith(a, "--threads="), ARGS)
    global const NUM_THREADS = i === nothing ? Threads.nthreads() : parse(Int, split(ARGS[i], "=")[2])
end

"""
    DEVICE / ON_GPU

`--device=cpu` (default) or `--device=cuda`. On `cuda` every toolkit that can reconstruct on an
NVIDIA GPU does so (`gpu_supports` in `_toolkits.jl`), each from host data to a host image: the
transfers to and from the device are inside the timed region, for every toolkit alike, since BART
cannot be timed any other way.

A GPU run has one host thread and OpenBLAS as the host BLAS, and refuses to start otherwise. One
thread because the device does the work, and because MRIReco's GPU path races with more: its operators
issue kernels from parallel OhMyThreads tasks, each on its own CUDA stream with nothing ordering
them, and on the 2D 8-coil Cartesian case its CG-SENSE returned NRMSE 0.94-1.29, different on every
call, against 0.675 on the CPU (NaN on the radial and L1-wavelet rows); at one thread every row
matched the CPU to 1e-6.
"""
const DEVICE = let i = findfirst(a -> startswith(a, "--device="), ARGS)
    d = i === nothing ? "cpu" : ARGS[i][(length("--device=") + 1):end]
    d in ("cpu", "cuda") || error("--device=$d: expected cpu or cuda")
    Symbol(d)
end
const ON_GPU = DEVICE === :cuda
ON_GPU && USE_MKL && error("--device=cuda runs with OpenBLAS as the host BLAS: drop --use-mkl")
ON_GPU && (NUM_THREADS != 1 || Threads.nthreads() != 1) &&
    error("--device=cuda runs with one host thread: pass -t 1 --threads=1 (see `DEVICE`)")

# Machine paths (BART builds, SigPy's interpreter, the data cache) come from the environment, filled
# from the untracked `benchmark/slurm/site.env` for anything not already set.
include(joinpath(@__DIR__, "..", "..", "utils", "config.jl"))
load_site_env!()

# CUDA.jl comes from a GPUEnv overlay of this environment rather than from its own dependencies, so
# a CPU run neither resolves nor loads it. Loading it also loads Ristretto's, NFFT's and
# RegularizedLeastSquares' GPU extensions.
if ON_GPU
    using GPUEnv
    GPUEnv.activate(; include_jlarrays = false, only_first = true, persist = true)
    using CUDA
    CUDA.functional() || error("--device=cuda but CUDA is not functional on $(gethostname())")
end

# Which BART build to time against: one per BLAS backend, and one for the GPU. With no build
# configured for this run, BART is left out of every section (see `should_run_framework`).
if USE_MKL
    @info "Enabling Intel MKL backend via MKL.jl"
    using MKL
end
const BART_KEY = ON_GPU ? "RISTRETTO_BENCH_BART_CUDA" : USE_MKL ? "RISTRETTO_BENCH_BART_MKL" : "RISTRETTO_BENCH_BART_OPENBLAS"
const BART_BINARY = get(ENV, BART_KEY, "")
const BART_AVAILABLE = !isempty(BART_BINARY) && isfile(BART_BINARY)
BART_AVAILABLE || @warn "No BART build configured for this backend, so BART rows are skipped" key = BART_KEY value = BART_BINARY

"""
    BACKEND

What the run is recorded under (`record_run`): `"cuda"`, `"mkl"` or `"openblas"`, and the suffix of
every framework label.
"""
const BACKEND = ON_GPU ? "cuda" : USE_MKL ? "mkl" : "openblas"
const BACKEND_LABEL = ON_GPU ? "CUDA" : USE_MKL ? "MKL" : "OpenBLAS"
const FW = "Ristretto ($BACKEND_LABEL)"
const BART_FW = get(ENV, "RISTRETTO_BENCH_BART_MEASURE", "0") == "1" ? "BART ($BACKEND_LABEL, MEASURE)" : "BART ($BACKEND_LABEL)"

using ThreadPinning
# Threads go only to allowed CPUs that are not SMT siblings: a sibling shares its core with another
# thread of the same run, which halves that core for both.
let mask = getaffinity()
    allowed = findall(==(1), mask) .- 1
    isempty(allowed) && (allowed = collect(0:(Threads.nthreads() - 1)))
    physical = filter(!ThreadPinning.ishyperthread, allowed)
    length(physical) >= Threads.nthreads() || error("only $(length(physical)) physical cores among the allowed CPUs $allowed, $(Threads.nthreads()) threads requested")
    global const PINNED_CPUS = physical[1:Threads.nthreads()]
end
pinthreads(PINNED_CPUS)
# No `mkl_set_dynamic(0)`: timed with and without it, MKL's own dynamic adjustment measured the same.
const CPU_STR = join(PINNED_CPUS, ",")
@info "Julia threads pinned" CPU_STR

BART_AVAILABLE && (ENV["TOOLBOX_PATH"] = BART_BINARY)
# BART plans its FFTs with FFTW_ESTIMATE unless `BART_USE_FFTW_WISDOM=1`, which plans the
# contiguous ones with FFTW_MEASURE and keeps the plans as wisdom files under
# `$BART_TOOLBOX_PATH/save/fftw/`. `RISTRETTO_BENCH_BART_MEASURE=1` times BART that way, as its own
# framework label, with the wisdom directory emptied before every run (`time_bart`), so each timed
# process pays its planning as the in-process toolkits do.
const BART_MEASURE = get(ENV, "RISTRETTO_BENCH_BART_MEASURE", "0") == "1"
const BART_WISDOM_DIR = BART_MEASURE ? mktempdir() : ""
ENV["BART_USE_FFTW_WISDOM"] = BART_MEASURE ? "1" : "0"
BART_MEASURE && (ENV["BART_TOOLBOX_PATH"] = BART_WISDOM_DIR)
function clear_bart_wisdom()
    BART_MEASURE || return nothing
    dir = joinpath(BART_WISDOM_DIR, "save", "fftw")
    rm(dir; force = true, recursive = true)
    mkpath(dir)
    return nothing
end
clear_bart_wisdom()
ENV["OMP_NUM_THREADS"] = string(NUM_THREADS)
ENV["OPENBLAS_NUM_THREADS"] = string(NUM_THREADS)
ENV["MKL_NUM_THREADS"] = string(NUM_THREADS)
ENV["GOMP_CPU_AFFINITY"] = CPU_STR
ENV["KMP_AFFINITY"] = "granularity=fine,proclist=[$CPU_STR],explicit"
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_PLACES"] = "{$CPU_STR}"

using Ristretto
using Ristretto: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo
using GeometricMedicalPhantoms
using LinearAlgebra
using Statistics
using Random
using FFTW
using BartIO
# SigPy and MRpro run in-process through PythonCall, on the interpreter `RISTRETTO_BENCH_SIGPY_PYTHON`
# names rather than an environment of CondaPkg's own. PythonCall picks its interpreter when it
# loads, so both variables are set before it does. It frees a Python object that Julia's GC
# finalizes on a thread without the GIL later, on the thread that holds it, which is what makes
# the multithreaded Ristretto solves between Python calls safe.
get!(ENV, "JULIA_CONDAPKG_BACKEND", "Null")
# A GPU run takes the interpreter `RISTRETTO_BENCH_GPU_PYTHON` names when one is set: the GPU builds of
# PyTorch and CuPy are an environment of their own, separate from the CPU-only one.
let py = get(ENV, ON_GPU && haskey(ENV, "RISTRETTO_BENCH_GPU_PYTHON") ? "RISTRETTO_BENCH_GPU_PYTHON" : "RISTRETTO_BENCH_SIGPY_PYTHON", "")
    isempty(py) || (ENV["JULIA_PYTHONCALL_EXE"] = py)
end
using PythonCall
using MRIReco

"""
    MRPRO_AVAILABLE

Whether MRpro imports in the Python interpreter, and on a GPU run whether its PyTorch sees a CUDA
device; its rows are skipped otherwise (see `should_run_framework`). PyTorch's intra-op pool is set
to `NUM_THREADS` and its inter-op pool to one thread before any tensor work; finufft, MRpro's
NUFFT, follows `OMP_NUM_THREADS` above.
"""
const MRPRO_AVAILABLE = try
    torch = pyimport("torch")
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(1)
    pyimport("mrpro")
    ON_GPU && !pyconvert(Bool, torch.cuda.is_available()) && error("PyTorch $(torch.__version__) sees no CUDA device")
    true
catch err
    @warn "MRpro is unavailable in the Python interpreter, so MRpro rows are skipped" exception = err
    false
end

"""
    SIGPY_AVAILABLE

Whether SigPy can run this backend: always on the CPU, and on a GPU run only when CuPy imports and
SigPy has enabled it.
"""
const SIGPY_AVAILABLE = !ON_GPU || try
    pyimport("cupy")
    pyconvert(Bool, pyimport("sigpy").config.cupy_enabled) || error("SigPy did not enable CuPy")
    true
catch err
    @warn "SigPy cannot use the GPU, so SigPy rows are skipped" exception = err
    false
end

"""
    MRIRECO_BLAS_THREADS

The BLAS thread count MRIReco chose for itself, captured before this file overrides it.

`MRIReco.__init__` sets `BLAS.set_num_threads(1)` when `Threads.nthreads() > 1`
(MRIReco.jl:20-26 — the other branch is Windows-only). That is a deliberate choice by the
package under test, and the harness must not silently undo it: setting the count here, *after*
`using MRIReco`, left every timed MRIReco row at `NUM_THREADS` while Ristretto pinned BLAS inside its
own solve, so the two toolkits were compared under different BLAS policies for no reason other
than the order of two lines in this file.

At `-t 1` on Linux MRIReco makes no choice at all, and OpenBLAS stays at its
`jl_effective_threads`-derived default — most of the node. That is not a policy to respect but a
known artifact: RegularizedLeastSquares makes ~30 BLAS-1 calls per ADMM outer iteration
(`norm`/`dot`/`rmul!` in `cg.jl` and the residual block), each spawning a full thread team over a
~9k-element vector, measured at 346 s for a TV solve that takes 1.08 s with BLAS pinned. So the
single-threaded case still gets `NUM_THREADS` (which is 1 there anyway).
"""
const MRIRECO_BLAS_THREADS = Threads.nthreads() > 1 ? BLAS.get_num_threads() : NUM_THREADS

# Everything else runs at the thread count under test. `with_mrireco_blas` puts MRIReco's own
# choice back for the duration of an MRIReco call, and restores this afterwards.
BLAS.set_num_threads(NUM_THREADS)
FFTW.set_num_threads(NUM_THREADS)
@info "BLAS/FFTW pinned" blas_threads = BLAS.get_num_threads() fftw_threads = FFTW.get_num_threads() mrireco_blas_threads = MRIRECO_BLAS_THREADS

"""
    check_environment()

Fail loudly instead of silently benchmarking under an environment that does not match what was
requested. It is an assertion, run automatically by every section, because a mismatch silently
invalidates the numbers rather than crashing anything -- exactly the kind of thing this suite
exists to catch in a toolkit, not commit on its own.
"""
function check_environment()
    cpus_allowed = let line = ""
        for l in eachline("/proc/self/status")
            startswith(l, "Cpus_allowed_list:") && (line = strip(split(l, ":")[2]))
        end
        line
    end
    n_allowed = sum(
        r -> (p = split(r, "-"); length(p) == 1 ? 1 : parse(Int, p[2]) - parse(Int, p[1]) + 1),
        split(cpus_allowed, ","),
    )
    n_allowed < NUM_THREADS && error(
        "requested $NUM_THREADS threads but only $n_allowed CPUs are allowed " *
            "(Cpus_allowed_list=$cpus_allowed) -- the SLURM allocation does not cover what was " *
            "asked for; rerun with a matching --cpus-per-task",
    )
    Threads.nthreads() != NUM_THREADS && error(
        "requested $NUM_THREADS threads but Julia started with $(Threads.nthreads()) -- pass -t $NUM_THREADS",
    )
    BLAS.get_num_threads() != NUM_THREADS && error(
        "BLAS is pinned to $(BLAS.get_num_threads()) threads, not the requested $NUM_THREADS",
    )
    USE_MKL && get(ENV, "KMP_BLOCKTIME", "") != "0" && error(
        "MKL is enabled but KMP_BLOCKTIME is $(get(ENV, "KMP_BLOCKTIME", "unset")), not \"0\" -- " *
            "export it before starting Julia (see docs/src/high-level/performance.md); MKL's " *
            "worker threads will otherwise spin and crowd out the ones being measured",
    )
    return nothing
end
check_environment()

"""
    with_mrireco_blas(f)

Run `f()` with BLAS at [`MRIRECO_BLAS_THREADS`](@ref) — what MRIReco set for itself — and restore
`NUM_THREADS` afterwards, including on exception.

Every timed MRIReco call goes through this, so MRIReco is measured under its own threading policy
and Ristretto under its own, rather than both under whichever one happened to be set last.
"""
function with_mrireco_blas(f)
    MRIRECO_BLAS_THREADS == NUM_THREADS && return f()
    BLAS.set_num_threads(MRIRECO_BLAS_THREADS)
    try
        return f()
    finally
        BLAS.set_num_threads(NUM_THREADS)
    end
end

include(joinpath(@__DIR__, "..", "src", "ComparisonHarness.jl"))
using .ComparisonHarness: check_nrmse, run_bart, run_bart_timed

# The case catalog, Ristretto's reconstruction of each method, `time_run` and the result store, shared
# with the Ristretto harness (benchmark/run.jl). No section prepares data of its own.
include(joinpath(@__DIR__, "..", "..", "utils", "bench_utils.jl"))
using .BenchUtils

const sigpy = pyimport("sigpy")
const sp_mri = pyimport("sigpy.mri")
const sp_app = pyimport("sigpy.mri.app")

@info "comparison setup" host = gethostname() julia = VERSION threads = Threads.nthreads() blas = BLAS.get_config().loaded_libs[1].libname mkl = USE_MKL bart = BART_BINARY

# Bare process spawn cost (`bart version` does no file I/O) — the floor for an input-less call.
const BART_SPAWN = BART_AVAILABLE ? let times = Float64[]
        for _ in 1:10
            t0 = time_ns()
            read(pipeline(ignorestatus(`$BART_BINARY version`)), String)
            push!(times, (time_ns() - t0) / 1.0e9)
    end
        minimum(times)
end : NaN
BART_AVAILABLE && @info @sprintf("BART spawn cost: %.1f ms", BART_SPAWN * 1000)

"""
    BART_GPU_INIT -> seconds

What a `pics -g` process pays to start using the GPU before it solves anything: creating the CUDA
context and loading cuFFT and cuBLAS. Every BART call is a fresh process and pays it again, while
an in-process toolkit pays it once, in its warm-up. Measured as the difference between the fastest
`Total Time` of five `pics -g` and five `pics` calls on an 8×8 single-coil problem, one iteration
each, and subtracted from every BART GPU timing. 0 on a CPU run.
"""
const BART_GPU_INIT = (ON_GPU && BART_AVAILABLE) ? let
        k = ones(ComplexF32, 8, 8, 1, 1)
        s = ones(ComplexF32, 8, 8, 1, 1)
        best(cmd) = minimum(_ -> last(run_bart_timed(1, cmd, k, s)), 1:5)
        run_bart(1, "pics -g -S -w 1 -i 1", k, s)                 # the first context also JIT-loads
        max(0.0, best("pics -g -S -w 1 -i 1") - best("pics -S -w 1 -i 1"))
end : 0.0
ON_GPU && BART_AVAILABLE && @info @sprintf("BART GPU initialisation: %.1f ms", BART_GPU_INIT * 1000)

"""
    time_bart(cmd, inputs...; nout = 1, num_runs = 3) -> (t_min_s, t_med_s, result)

Run BART `cmd` (`pics`) on `inputs` `num_runs` times after a warm-up, each run's time being the
`Total Time` the tool reports itself (`run_bart_timed`), less [`BART_GPU_INIT`](@ref) when `cmd`
runs on the GPU (`-g`). That time starts once the process is loaded and ends with the image
written to its memory-mapped output, so it leaves out the process start and the harness' file
writes and reads, as the in-process toolkits' times leave out their loading, and it keeps the
mapping of the inputs into memory, as theirs keep the copies from and to the host. Every call is
a fresh `bart` process with no FFTW wisdom to read (`BART_USE_FFTW_WISDOM=0`, or an emptied
wisdom directory under `RISTRETTO_BENCH_BART_MEASURE=1`), so each timed run plans its FFTs from
scratch, as [`time_run`](@ref) makes the in-process toolkits do.
"""
function time_bart(cmd::AbstractString, inputs...; nout::Int = 1, num_runs::Int = RUNS[])
    if WARMUP[] == 0         # images only (calibration): one untimed-for-the-record run
        clear_bart_wisdom()
        res, t = run_bart_timed(nout, cmd, inputs...)
        return t, t, res
    end
    init = occursin(r"(^| )-g( |$)", cmd) ? BART_GPU_INIT : 0.0
    clear_bart_wisdom()
    res, _ = run_bart_timed(nout, cmd, inputs...)
    times = Float64[]
    for _ in 1:num_runs
        clear_bart_wisdom()
        res, t = run_bart_timed(nout, cmd, inputs...)
        push!(times, t - init)
    end
    return max(1.0e-5, minimum(times)), max(1.0e-5, median(times)), res
end

"""
    RUNS

Timed runs of the current case: 3, or 1 for a heavy case (`timed_runs`). `toolkit_run` sets it, and
[`time_reconstruction`](@ref) and [`time_bart`](@ref) default to it.
"""
const RUNS = Ref(3)

"""
    WARMUP

Untimed warm-up runs before the timed ones: 1, or 0 in `calibrate_lambda.jl`, which only needs
the images.
"""
const WARMUP = Ref(1)

"""
    time_reconstruction(f; num_runs = RUNS[]) -> (t_min_s, t_med_s, result)

`time_run` (`WARMUP[]` warm-ups, then the minimum and median of `num_runs`) for the in-process
toolkits, Ristretto's timing function in the harness too. BART goes through [`time_bart`](@ref) instead.
"""
time_reconstruction(f; num_runs::Int = RUNS[]) = time_run(f; warmup = WARMUP[], runs = num_runs)

"""
    BenchResult

One row: `category` is the section, `method` the method label, `case_id` the catalog case and
`data_source` where its data came from (`"synthetic"` or the real dataset).
"""
struct BenchResult
    category::String
    method::String
    framework::String
    threads::Int
    time_ms::Float64
    nrmse_gt::Float64
    nrmse_ristretto::Float64
    case_id::String
    data_source::String
end

results = BenchResult[]

"""
    CASE_FILTER

Parsed from `--cases=pat1,pat2,...`, or `nothing` when not passed (run everything). Each `pat` is a
case-insensitive substring matched against a catalog case id (`--cases=shepp_logan_2d` runs the
three 2D Shepp-Logan cases, `--cases=cine` both cine cases), a section, or a method label
(`--cases=low-rank` runs every low-rank row). Every section checks [`should_run_case`](@ref) and
[`should_run`](@ref) before paying for a solve, so a rerun of one suspect case does not have to pay
for the whole section.
"""
const CASE_FILTER = let i = findfirst(a -> startswith(a, "--cases="), ARGS)
    i === nothing ? nothing : [lowercase(s) for s in split(ARGS[i][(length("--cases=") + 1):end], ",")]
end

"""
    DATA

`--data=synthetic` (default), `real` or `all`: which catalog cases the sections iterate. Real data
are the real-data analogues of the synthetic cases (benchmark/utils/real_data.jl).
"""
const DATA = let i = findfirst(a -> startswith(a, "--data="), ARGS)
    d = i === nothing ? "synthetic" : ARGS[i][(length("--data=") + 1):end]
    d in ("synthetic", "real", "all") || error("--data=$d: expected synthetic, real or all")
    d
end

"""
    should_run(label, method) -> Bool

True unless [`CASE_FILTER`](@ref) is set and no pattern in it is a substring of `label` (a case id
or a section) or of `method` (case-insensitive).
"""
should_run(category, method) = CASE_FILTER === nothing ||
    any(p -> occursin(p, lowercase(category)) || occursin(p, lowercase(method)), CASE_FILTER)

"""
    should_run_case(id) -> Bool

Whether any row of case `id` can pass [`CASE_FILTER`](@ref): true when no filter is set, or when a
pattern does not name a case at all (then it filters by section or method instead).
"""
function should_run_case(id)
    CASE_FILTER === nothing && return true
    all_ids = lowercase.(case_ids(; real = true))
    return any(p -> occursin(p, lowercase(id)) || !any(i -> occursin(p, i), all_ids), CASE_FILTER)
end

"""
    FRAMEWORK_FILTER

Parsed from `--frameworks=pat1,pat2,...`, or `nothing`. Each `pat` is a case-insensitive substring
matched against a framework label (`"BART"` matches `"BART (MKL)"` and `"BART (OpenBLAS)"` alike).
Gates only the *competitor* toolkits (SigPy/BART/MRIReco/MIRT/MRpro) in each case, never Ristretto itself: Ristretto's
own solve is the reference every other framework's `nrmse_ristretto` is computed against, so it always
runs regardless of this filter, and stays cheap next to whichever toolkit is under suspicion.
"""
const FRAMEWORK_FILTER = let i = findfirst(a -> startswith(a, "--frameworks="), ARGS)
    i === nothing ? nothing : [lowercase(s) for s in split(ARGS[i][(length("--frameworks=") + 1):end], ",")]
end

"""
    should_run_framework(framework) -> Bool

True unless [`FRAMEWORK_FILTER`](@ref) is set and no pattern in it is a substring of `framework`
(case-insensitive), or `framework` is BART and no BART build is configured for this backend, or
MRpro and it does not import ([`MRPRO_AVAILABLE`](@ref)), or SigPy on a GPU run without CuPy
([`SIGPY_AVAILABLE`](@ref)). See [`FRAMEWORK_FILTER`](@ref) -- never
call this for Ristretto's own row.
"""
function should_run_framework(framework)
    occursin("bart", lowercase(framework)) && !BART_AVAILABLE && return false
    occursin("mrpro", lowercase(framework)) && !MRPRO_AVAILABLE && return false
    occursin("sigpy", lowercase(framework)) && !SIGPY_AVAILABLE && return false
    return FRAMEWORK_FILTER === nothing || any(p -> occursin(p, lowercase(framework)), FRAMEWORK_FILTER)
end

using .BenchUtils.ResultsStore: record_run

"""
    toolkit_versions() -> Dict{String, String}

The version of every toolkit this process can time, and of the Ristretto checkout (its commit).
"""
function toolkit_versions()
    v = Dict{String, String}()
    v["Ristretto"] = let dir = pkgdir(Ristretto)
        commit = strip(read(ignorestatus(`git -C $dir rev-parse --short HEAD`), String))
        dirty = !isempty(strip(read(ignorestatus(`git -C $dir status --porcelain --untracked-files=no -- src ext deps`), String)))
        string(pkgversion(Ristretto), " (", commit, dirty ? ", modified" : "", ")")
    end
    BART_AVAILABLE && (v["BART"] = strip(read(ignorestatus(`$BART_BINARY version`), String)))
    v["MRIReco"] = string(pkgversion(MRIReco))
    v["MIRT"] = string(pkgversion(ComparisonHarness.MIRTBridge.MIRT))
    for (name, mod) in (("SigPy", "sigpy"), ("MRpro", "mrpro"), ("PyTorch", "torch"), ("CuPy", "cupy"))
        try
            v[name] = pyconvert(String, pyimport(mod).__version__)
        catch
        end
    end
    return v
end

"""
    flush_results!(name) -> path or nothing

Write and clear whatever is currently in `results`, as its own immutable run file, right now.
Every section calls this after each case (see `should_run`'s guarded blocks in the `run_<section>.jl`
scripts) rather than only once at the end, so a crash partway through a long section -- a slow
real-data solve, a hung toolkit subprocess -- does not lose the cases that already finished. Returns
`nothing` when there is nothing to flush (already flushed, or the case was skipped).
"""
function flush_results!(name::AbstractString)
    isempty(results) && return nothing
    path = record_run(
        name, BACKEND, NUM_THREADS, results;
        hostname = gethostname(), cpu_model = Sys.cpu_info()[1].model, cpu_threads = Sys.CPU_THREADS,
        julia_version = string(VERSION),
        julia_threads = Threads.nthreads(), blas_vendor = BLAS.get_config().loaded_libs[1].libname,
        use_mkl = USE_MKL, bart_binary = BART_BINARY, pinned_cpus = CPU_STR,
        placement = get(ENV, "RISTRETTO_BENCH_PLACEMENT", "isolated"),
        bart_spawn_ms = BART_SPAWN * 1000,
        cases_filter = CASE_FILTER, frameworks_filter = FRAMEWORK_FILTER, data = DATA,
        small = small_mode(), cine_frames = cine_frames(), device = String(DEVICE),
        gpu = ON_GPU ? CUDA.name(CUDA.device()) : nothing,
        toolkit_versions = toolkit_versions(),
    )
    for r in results
        @printf(
            "%-38s | %-26s | %-22s | %3d | %10.2f ms | %10.2e | %10.2e\n",
            r.case_id, r.method, r.framework, r.threads, r.time_ms, r.nrmse_gt, r.nrmse_ristretto
        )
    end
    @info "flushed run" path n = length(results) source = ResultsStore.source_tag()
    empty!(results)
    return path
end

"""
    release_device_memory()

Return what the toolkits' memory pools hold on the GPU (CUDA.jl's, CuPy's, PyTorch's caching
allocator) to the device after each toolkit's row, so the next toolkit starts with the whole device
rather than with what the previous one kept cached. A no-op on a CPU run.
"""
function release_device_memory()
    ON_GPU || return nothing
    GC.gc()
    CUDA.reclaim()
    SIGPY_AVAILABLE && pyimport("cupy").get_default_memory_pool().free_all_blocks()
    MRPRO_AVAILABLE && pyimport("torch").cuda.empty_cache()
    return nothing
end

"""Final catch-all flush at the end of a section script -- a no-op if every case already flushed
itself via [`flush_results!`](@ref)."""
write_section(name::AbstractString) = flush_results!(name)

"""
    CMP_CTYPE / CMP_RTYPE

The complex (and matching real) element type every toolkit reconstructs in. **`ComplexF32`**, which
is what BART is: its `complex float` is a pair of `float32`, with no double-precision build option,
so a double-precision run of the other four compares a toolkit doing twice the memory traffic
against one that is not. Single precision is also what MRI reconstruction is done in — k-space off
the scanner is 16-bit integer or 32-bit float.

Set `CMP_PRECISION=double` to go back to `ComplexF64` for everything except BART, which cannot.

Only the *solve* runs in this type. Each bridge promotes its result to `ComplexF64` on the way out,
outside the timed region, so the NRMSE column is not itself computed at the precision under test.
"""
const CMP_CTYPE = get(ENV, "CMP_PRECISION", "single") == "double" ? ComplexF64 : ComplexF32
const CMP_RTYPE = real(CMP_CTYPE)
@info "comparison precision" ctype = CMP_CTYPE

# Every section's data comes from the case catalog (`get_case`), in `ComplexF32`. The sensitivity
# maps are handed to every toolkit exactly as the catalog produces them: `normalize_sensitivity_maps`
# is deliberately not called, since it would give Ristretto a known operator norm and so a free step size,
# while MRIReco's `SensitivityOp` and BART's `pics` normalize nothing.

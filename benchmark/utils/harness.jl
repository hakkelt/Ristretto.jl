# Timing and provenance for the Ristretto harness and the comparison suite.

# Ristretto keeps FFTW wisdom on disk between sessions (`Ristretto.fftw_wisdom_path`); a
# timed run would then reuse plans an earlier run measured, which none of the other toolkits do.
# Every timing here plans from scratch (see `time_run`), so the cache is off.
ENV["RISTRETTO_FFTW_WISDOM"] = "off"

"""
    time_run(f; warmup = 1, runs = 3, cold_fft = true) -> (min_s, median_s, result)

Call `f()` `warmup` times untimed (compilation, first-touch allocation), then `runs` times timed;
return the minimum and median wall time in seconds and the last result.

With `cold_fft`, FFTW's accumulated wisdom is dropped before every call, so each timed run plans
its FFTs from scratch as a fresh process would. FFTW keeps what a `MEASURE` plan learned for the
lifetime of the process and answers a later plan of the same size from it in microseconds, which
would otherwise take a 128³ × 8-coil `MEASURE` plan from about 1 s in the warm-up to 0.1 ms in
every timed run, and would also hand an `ESTIMATE` plan the measured one. The wisdom is
process-wide, so this covers every in-process toolkit on FFTW.jl (Ristretto, MIRT, MRIReco).

BenchmarkTools' sampling is built for microsecond kernels; these are reconstructions of seconds to
minutes, where a handful of runs and their minimum is the robust estimate (noise on a shared node
only ever adds time). The median is kept alongside, as the spread (`median / min`) says whether the
minimum can be trusted.
"""
function time_run(f; warmup::Integer = 1, runs::Integer = 3, cold_fft::Bool = true)
    runs >= 1 || throw(ArgumentError("runs must be at least 1"))
    res = nothing
    for _ in 1:warmup
        cold_fft && FFTW.forget_wisdom()
        res = f()
    end
    times = Float64[]
    for _ in 1:runs
        cold_fft && FFTW.forget_wisdom()
        t0 = time_ns()
        res = f()
        push!(times, (time_ns() - t0) / 1.0e9)
    end
    return minimum(times), median(times), res
end

"""
    timed_runs(c::BenchCase) -> Int

Timed runs for case `c`: 3, or 1 for a heavy case (the 3D volume and the cine series), whose single
reconstruction takes long enough that its run-to-run spread is small against it.
"""
timed_runs(c::BenchCase) = c.heavy ? 1 : 3

_git(dir, args...) = try
    strip(read(pipeline(`git -C $dir $args`; stderr = devnull), String))
catch
    ""
end

"""
    git_ref(dir) -> NamedTuple (commit, branch, dirty)

The commit checked out in `dir`, its branch, and whether tracked files differ from it.
"""
function git_ref(dir::AbstractString)
    commit = _git(dir, "rev-parse", "HEAD")
    branch = _git(dir, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = !isempty(_git(dir, "status", "--porcelain", "--untracked-files=no"))
    return (; commit, branch, dirty)
end

"""
    tree_hash(dir, path = ".") -> String

The git tree hash of `path` in `dir`'s checkout, or `"dirty-<sha1 of the diff>"` when `path` has
uncommitted changes, or `""` outside git. Two runs with the same tree hash ran the same code.
"""
function tree_hash(dir::AbstractString, path::AbstractString = ".")
    h = _git(dir, "rev-parse", path == "." ? "HEAD^{tree}" : "HEAD:$path")
    isempty(h) && return ""
    diff = _git(dir, "diff", "HEAD", "--", path)
    return isempty(diff) ? h : "dirty-" * bytes2hex(sha1(diff))[1:12]
end

"""
    node_class() -> String

What the timings are comparable across: the CPU model plus the SLURM partition (`login` outside a
job). Results from different node classes are never treated as the same measurement.

A run that shared its NUMA domain with other tasks (`matrix.sh --pack`, which sets
`RISTRETTO_BENCH_PLACEMENT=shared`) competed with them for L3 and memory bandwidth, so it gets a class of
its own: an isolated baseline is never skipped because a packed one is stored, and `compare.jl`
warns when it compares the two.
"""
function node_class()
    cpu = "unknown cpu"
    try
        for l in eachline("/proc/cpuinfo")
            if startswith(l, "model name")
                cpu = strip(split(l, ":"; limit = 2)[2])
                break
            end
        end
    catch
        # Windows has no /proc; Sys.cpu_info gives the same model string from the registry
        cpu = strip(Sys.cpu_info()[1].model)
    end
    class = string(cpu, " / ", get(ENV, "SLURM_JOB_PARTITION", "login"))
    return get(ENV, "RISTRETTO_BENCH_PLACEMENT", "isolated") == "shared" ? class * " / shared domain" : class
end

"""
    RECORDED_ENV

Environment variables that change what a benchmark measures, recorded with every result.
"""
const RECORDED_ENV = (
    "KMP_BLOCKTIME", "OPENBLAS_THREAD_TIMEOUT", "OMP_WAIT_POLICY", "MKL_DYNAMIC", "TMPDIR",
    "JULIA_EXCLUSIVE", "RISTRETTO_BENCH_SNR_DB", "CMP_OUTER", "CMP_CG_ITERS", "RISTRETTO_BENCH_PLACEMENT",
)

recorded_env() = Dict(k => get(ENV, k, "") for k in RECORDED_ENV)

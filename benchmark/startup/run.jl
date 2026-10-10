# Startup cost of every toolkit of the cross-toolkit comparison, each measured in fresh processes:
#
#   julia --project=benchmark/comparison benchmark/startup/run.jl
#
# The comparison tables report warm solves (the fastest of three after a warm-up), which leaves out
# what a user pays before the first image: starting the runtime, loading the package and, for the
# Julia toolkits, compiling the solve on its first call. This script measures those on the
# `shepp_logan_2d_8ch_cartesian` case (128², 8 coils), for the L1-wavelet row (20 iterations, every
# toolkit but MIRT, which has no wavelet prox) and the CG-SENSE row (10 iterations, every toolkit):
#
#   * `runtime`: wall time of a process that does nothing (`julia -e 0`, `python -c pass`);
#   * `import`: wall time of a process that only loads the toolkit (`using Ristretto`,
#     `import sigpy`, ...; for BART, `bart version`);
#   * `first` / `warm`: in a fresh process, the first solve (compilation included) and the
#     median of `STARTUP_NWARM` further solves;
#   * `end_to_end`: wall time of a fresh process that loads the toolkit, reads the case and solves
#     it once. For BART every solve is such a process (`bart pics`), so its first, warm and
#     end-to-end times are one number.
#   * `precompile`: once, the time `Base.compilecache` takes to precompile Ristretto into an empty
#     depot (its dependencies' caches already exist). Every other Julia number assumes Ristretto,
#     MRIReco and MIRT are precompiled.
#
# Every child runs on the same `STARTUP_THREADS` physical cores of one NUMA domain (`numactl`),
# with OpenMP, OpenBLAS, MKL and PyTorch at that thread count, Julia at `-t STARTUP_THREADS`, and
# BLAS and FFTW as the comparison harness sets them (`_setup.jl`). Toolkits are interleaved within
# each repetition. Run it through `benchmark/startup/startup.sh` on a compute node.
#
# Environment: STARTUP_THREADS (8), STARTUP_REPS (5, fresh processes per measurement),
# STARTUP_WARM_REPS (3, fresh processes timing first and warm solves), STARTUP_NWARM (10),
# STARTUP_DATADIR (`tempdir()/ristretto-startup`), STARTUP_OUT (`benchmark/startup/results.json`).
# Machine paths (the BART build, the Python interpreter) come from `benchmark/slurm/site.env`.

using Dates, JSON, Printf, Statistics, TOML

include(joinpath(@__DIR__, "..", "utils", "config.jl"))
load_site_env!()

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const PROJECT = joinpath(REPO, "benchmark", "comparison")
const THREADS = parse(Int, get(ENV, "STARTUP_THREADS", "8"))
const REPS = parse(Int, get(ENV, "STARTUP_REPS", "5"))
const WARM_REPS = parse(Int, get(ENV, "STARTUP_WARM_REPS", "3"))
const NWARM = parse(Int, get(ENV, "STARTUP_NWARM", "10"))
const DATADIR = get(ENV, "STARTUP_DATADIR", joinpath(tempdir(), "ristretto-startup"))
const OUT = get(ENV, "STARTUP_OUT", joinpath(@__DIR__, "results.json"))
const CASE = "shepp_logan_2d_8ch_cartesian"
const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())
const PYTHON = get(ENV, "RISTRETTO_BENCH_SIGPY_PYTHON", "")
const BART = get(ENV, "RISTRETTO_BENCH_BART_OPENBLAS", "")

# ---------------------------------------------------------------- placement

"""
    pick_cpus(n) -> (cpus, node)

`n` allowed CPUs on distinct physical cores of one NUMA node (the first one that has enough).
"""
function pick_cpus(n)
    allowed = Set{Int}()
    for l in eachline("/proc/self/status")
        startswith(l, "Cpus_allowed_list:") || continue
        for r in split(strip(split(l, ":")[2]), ",")
            p = parse.(Int, split(r, "-"))
            union!(allowed, p[1]:p[end])
        end
    end
    seen = Set{Int}()
    bynode = Dict{Int, Vector{Int}}()
    for l in eachline(`lscpu -p=CPU,CORE,NODE`)
        startswith(l, "#") && continue
        cpu, core, node = parse.(Int, split(l, ","))
        (cpu in allowed && !(core in seen)) || continue
        push!(seen, core)
        push!(get!(bynode, node, Int[]), cpu)
    end
    for node in sort!(collect(keys(bynode)))
        length(bynode[node]) >= n && return bynode[node][1:n], node
    end
    error("no NUMA node has $n allowed physical cores: $bynode")
end

const CPUS, NODE = pick_cpus(THREADS)
const CPU_STR = join(CPUS, ",")
const PIN = Sys.which("numactl") === nothing ? `taskset -c $CPU_STR` : `numactl --physcpubind=$CPU_STR --membind=$NODE`
const CHILD_ENV = merge(
    filter(p -> !startswith(p.first, "JULIA_NUM_THREADS"), Dict(ENV)),
    Dict(
        "OMP_NUM_THREADS" => string(THREADS), "OPENBLAS_NUM_THREADS" => string(THREADS),
        "MKL_NUM_THREADS" => string(THREADS), "OMP_PROC_BIND" => "close", "OMP_PLACES" => "{$CPU_STR}",
        "GOMP_CPU_AFFINITY" => CPU_STR, "BART_USE_FFTW_WISDOM" => "0", "PYTHONWARNINGS" => "ignore",
    ),
)

const JL = `$JULIA --startup-file=no --project=$PROJECT -t $THREADS`

"""
    timed(cmd) -> (wall_s, stdout)

Run `cmd` pinned, in the child environment, and return its wall time and output.
"""
function timed(cmd; dir = DATADIR)
    out = IOBuffer()
    err = IOBuffer()
    t0 = time_ns()
    p = run(pipeline(ignorestatus(setenv(`$PIN $cmd`, CHILD_ENV; dir)); stdout = out, stderr = err))
    t = (time_ns() - t0) / 1.0e9
    success(p) || error("$cmd failed:\n$(String(take!(err)))")
    return t, String(take!(out))
end

# ---------------------------------------------------------------- the case

isfile(joinpath(DATADIR, "meta.json")) ||
    run(`$JULIA --startup-file=no --project=$PROJECT $(joinpath(@__DIR__, "prepare.jl")) $DATADIR $CASE`)
const META = JSON.parsefile(joinpath(DATADIR, "meta.json"))
META["wavelet_levels"] == 3 || error("child_common.jl assumes 3 wavelet levels, the harness uses $(META["wavelet_levels"])")
const NX, NY, NC = META["nx"], META["ny"], META["coils"]
const MAXIT = Dict("wavelet" => META["wavelet_iterations"], "cgsense" => META["cgsense_iterations"])
const REFERENCE = read!(joinpath(DATADIR, "reference.c64"), Array{ComplexF32}(undef, NX, NY))

function mag_nrmse(est, ref)
    a, r = abs.(est), abs.(ref)
    a .*= sqrt(sum(abs2, r) / sum(abs2, a))
    return sqrt(sum(abs2, a .- r) / sum(abs2, r))
end

# ---------------------------------------------------------------- the toolkits

const TOOLKITS = [
    (name = "Ristretto", key = "Ristretto", lang = :julia, script = "solve_ristretto.jl", imp = "using Ristretto", methods = ("wavelet", "cgsense")),
    (name = "MRIReco", key = "MRIReco", lang = :julia, script = "solve_mrireco.jl", imp = "using MRIReco", methods = ("wavelet", "cgsense")),
    (name = "MIRT", key = "MIRT", lang = :julia, script = "solve_mirt.jl", imp = "using MIRT", methods = ("cgsense",)),
    (name = "SigPy", key = "SigPy", lang = :python, script = "solve_sigpy.py", imp = "import sigpy, sigpy.mri.app", methods = ("wavelet", "cgsense")),
    (name = "MRpro", key = "MRpro", lang = :python, script = "solve_mrpro.py", imp = "import mrpro", methods = ("wavelet", "cgsense")),
    (name = "BART", key = "BART", lang = :bart, script = "", imp = "", methods = ("wavelet", "cgsense")),
]

function available(tk)
    tk.lang === :python && return isfile(PYTHON)
    tk.lang === :bart && return isfile(BART)
    return true
end
const ACTIVE = filter(available, TOOLKITS)
for tk in TOOLKITS
    tk in ACTIVE || @warn "$(tk.name) is not configured on this machine (benchmark/slurm/site.env), skipped"
end

λ(tk, m) = m == "cgsense" ? 0.0 : Float64(META["lambda_wavelet"][tk.key])

import_cmd(tk) = tk.lang === :julia ? `$JL -e $(tk.imp)` : tk.lang === :python ? `$PYTHON -c $(tk.imp)` : `$BART version`

function solve_cmd(tk, m, nwarm)
    args = [DATADIR, m, string(nwarm), string(NX), string(NY), string(NC), string(MAXIT[m]), string(λ(tk, m))]
    script = joinpath(@__DIR__, tk.script)
    return tk.lang === :julia ? `$JL $script $args` : `$PYTHON $script $args`
end

bart_pics(m) = m == "wavelet" ?
    `$BART pics -S -w 1 -e -i $(MAXIT[m]) -R W:3:0:$(λ((key = "BART",), m)) ksp sens out_$m` :
    `$BART pics -S -w 1 -i $(MAXIT[m]) ksp sens out_$m`

function parse_result(out)
    m = match(r"RESULT (\S+) (\S+) (\S+) (\S+) (\S+)", out)
    m === nothing && error("no RESULT line in\n$out")
    warm = m[5] == "-" ? Float64[] : parse.(Float64, split(m[5], ","))
    return (import_s = parse(Float64, m[1]), setup_s = parse(Float64, m[2]), first_s = parse(Float64, m[3]), nrmse = parse(Float64, m[4]), warm_s = warm)
end

# ---------------------------------------------------------------- measuring

raw = Dict{String, Any}()
rec!(keys...; value) = push!(get!(raw, join(keys, "/"), Any[]), value)

# One run of each command first, untimed: page cache, and a precompile check of the Julia envs.
println("warming the file system cache...")
timed(`$JL -e 'using Ristretto, MRIReco, MIRT'`)
for tk in ACTIVE
    timed(import_cmd(tk))
end

# Precompiling Ristretto into a fresh depot ahead of the real one: `compilecache` writes the new
# cache there and finds the dependencies' caches in the real depot. `STARTUP_PRECOMPILE=0` skips it.
precompile_s = get(ENV, "STARTUP_PRECOMPILE", "1") == "0" ? nothing : let depot = mktempdir(DATADIR)
        env = merge(CHILD_ENV, Dict("JULIA_DEPOT_PATH" => depot * ":" * join(DEPOT_PATH, ":")))
        code = "t = @elapsed Base.compilecache(Base.identify_package(\"Ristretto\")); println(\"PRECOMPILE \", t)"
        out = read(setenv(`$PIN $JL -e $code`, env; dir = DATADIR), String)
        rm(depot; recursive = true, force = true)
        t = parse(Float64, match(r"PRECOMPILE (\S+)", out)[1])
        @printf("Ristretto precompile into an empty depot: %.1f s\n", t)
        t
end

for rep in 1:REPS
    println("repetition $rep/$REPS")
    rec!("runtime", "Julia"; value = first(timed(`$JL -e 0`)))
    isfile(PYTHON) && rec!("runtime", "Python"; value = first(timed(`$PYTHON -c pass`)))
    for tk in ACTIVE
        rec!("import", tk.name; value = first(timed(import_cmd(tk))))
        for m in tk.methods
            if tk.lang === :bart
                t, _ = timed(bart_pics(m))
                rec!("end_to_end", tk.name, m; value = t)
                continue
            end
            t, out = timed(solve_cmd(tk, m, 0))
            r = parse_result(out)
            rec!("end_to_end", tk.name, m; value = t)
            rec!("first", tk.name, m; value = r.first_s)
            rec!("nrmse", tk.name, m; value = r.nrmse)
            rec!("import_in_process", tk.name; value = r.import_s)
            if rep <= WARM_REPS
                _, out = timed(solve_cmd(tk, m, NWARM))
                r = parse_result(out)
                rec!("first", tk.name, m; value = r.first_s)
                rec!("warm", tk.name, m; value = r.warm_s)
            end
        end
    end
end
for m in ("wavelet", "cgsense")
    isfile(BART) || break
    out = read!(joinpath(DATADIR, "out_$m.cfl"), Array{ComplexF32}(undef, NX, NY))
    rec!("nrmse", "BART", m; value = mag_nrmse(out, REFERENCE))
end

# ---------------------------------------------------------------- summary

med(k) = haskey(raw, k) ? median(reduce(vcat, raw[k])) : nothing
summary = Any[]
for tk in ACTIVE, m in tk.methods
    e2e = med("end_to_end/$(tk.name)/$m")
    bart = tk.lang === :bart
    push!(
        summary, Dict(
            "toolkit" => tk.name, "method" => m,
            "runtime_s" => bart ? nothing : med("runtime/$(tk.lang === :julia ? "Julia" : "Python")"),
            "import_s" => med("import/$(tk.name)"),
            "first_solve_s" => bart ? e2e : med("first/$(tk.name)/$m"),
            "warm_solve_s" => bart ? e2e : med("warm/$(tk.name)/$m"),
            "end_to_end_s" => e2e,
            "nrmse" => med("nrmse/$(tk.name)/$m"),
        )
    )
end

fmt(x) = x === nothing ? "—" : @sprintf("%.3f", x)
println()
println("| toolkit | method | import s | first solve s | warm solve s | end to end s | NRMSE |")
println("|---|---|--:|--:|--:|--:|--:|")
for s in summary
    println("| $(s["toolkit"]) | $(s["method"]) | $(fmt(s["import_s"])) | $(fmt(s["first_solve_s"])) | $(fmt(s["warm_solve_s"])) | $(fmt(s["end_to_end_s"])) | $(fmt(s["nrmse"])) |")
end

lscpu = Dict(strip(a) => strip(b) for (a, b) in (split(l, ":"; limit = 2) for l in eachline(`lscpu`) if occursin(":", l)))
manifest = TOML.parsefile(joinpath(PROJECT, "Manifest.toml"))["deps"]
pyversions = isfile(PYTHON) ? strip(read(setenv(`$PYTHON -c "import sys, sigpy, mrpro, torch, numpy; print(sys.version.split()[0], sigpy.__version__, mrpro.__version__, torch.__version__, numpy.__version__)"`, CHILD_ENV), String)) : ""
gitref = strip(read(Cmd(`git rev-parse --short HEAD`; dir = REPO), String))

result = Dict(
    "case" => CASE,
    "date" => string(now()),
    "host" => gethostname(),
    "cpu_model" => get(lscpu, "Model name", ""),
    "threads" => THREADS,
    "cpus" => CPUS,
    "numa_node" => NODE,
    "pinning" => string(PIN),
    "julia_version" => string(VERSION),
    "ristretto_commit" => gitref,
    "julia_packages" => Dict(p => get(manifest[p][1], "version", "dev") for p in ("MRIReco", "MIRT") if haskey(manifest, p)),
    "python" => pyversions == "" ? nothing : Dict(zip(("python", "sigpy", "mrpro", "torch", "numpy"), split(pyversions))),
    "bart_version" => isfile(BART) ? strip(read(`$BART version`, String)) : nothing,
    "reps" => REPS, "warm_reps" => WARM_REPS, "nwarm" => NWARM,
    "iterations" => MAXIT,
    "ristretto_precompile_s" => precompile_s,
    "summary" => summary,
    "raw" => raw,
)
open(io -> JSON.print(io, result, 2), OUT, "w")
println("\nwrote $OUT")

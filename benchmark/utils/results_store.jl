# Concurrency-safe result storage: every recorded run is its own immutable file under
# `results/runs/`, never overwritten. That is the whole design -- a shared mutable file (the old
# `write_section` one-file-per-section-per-config JSON, or a single SQLite database on the same
# NFS-backed /project mount) has exactly the failure mode this replaces: two writers on different
# SLURM nodes (or a login-node smoke test run alongside a cluster job) can race on the same path,
# and the loser's data is gone with no trace it ever ran. A brand-new, uniquely-named file per run
# cannot collide with anything, on any filesystem, with no locking required.
#
# Querying is a read-time concern, not a write-time one: `query_results.jl` globs `results/runs/`
# and filters/sorts the rows in plain Julia -- no database involved at all, so there is nothing to
# keep in sync with the flat files, which stay the only source of truth.
#
# Two stores use this layout: the comparison suite's (`benchmark/comparison/results/runs/`, rows of
# `category`/`method`/`framework`) and the Ristretto harness's (`benchmark/results/runs/`, rows keyed by
# catalog case and method, with the git provenance of the checkout measured). Every function takes
# the store's `root`; the default is the comparison suite's, which is what its scripts expect.
module ResultsStore

using JSON, Dates

"""
    SCHEMA_VERSION

Written into every run file. 1 (or absent): rows from before the case catalog, whose inputs are
not the catalog's cases; 2: rows measured on catalog cases.
"""
const SCHEMA_VERSION = 2

const COMPARISON_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "comparison", "results"))
const HARNESS_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))
const RESULTS_DIR = COMPARISON_RESULTS_DIR
const RUNS_DIR = joinpath(RESULTS_DIR, "runs")
runs_dir(root::AbstractString) = joinpath(root, "runs")

"""
    source_tag(hostname = gethostname()) -> "slurm" | "other"

SLURM compute nodes on this cluster are named `x<digits>c...`; the login node and anything else
(a laptop, a different cluster) is "other". Queries default to preferring `"slurm"` rows -- a
login-node run is real data (useful for a quick smoke test or an isolated repro) but was not taken
under the pinned/exclusive conditions the committed comparison numbers assume.
"""
source_tag(hostname::AbstractString = gethostname()) = occursin(r"^x\d+c", hostname) ? "slurm" : "other"

"""
    record_run(section, backend, threads, rows; kwargs...) -> path

Write one immutable JSON file recording this process's rows for `section` at (`backend`,
`threads`). `rows` is a `Vector{BenchResult}` (see `_setup.jl`). `kwargs` are extra metadata
(hostname, julia_version, blas_vendor, bart_binary, pinned_cpus, cases_filter, ...)
stored alongside, exactly as the old per-section JSON did.

The filename encodes everything a query needs to filter without opening the file: timestamp,
backend, threads, section, source, pid. `ispath` is checked and a random suffix appended on the
(astronomically unlikely) chance two runs land in the same millisecond under the same pid.
"""
function record_run(section::AbstractString, backend::AbstractString, threads::Integer, rows; kwargs...)
    benchmarks = [
        Dict(
            "category" => r.category, "method" => r.method, "framework" => r.framework,
            "threads" => r.threads, "time_ms" => _json_num(r.time_ms),
            "nrmse_gt" => _json_num(r.nrmse_gt), "nrmse_ristretto" => _json_num(r.nrmse_ristretto),
            "case_id" => hasproperty(r, :case_id) ? r.case_id : "",
            "data_source" => hasproperty(r, :data_source) ? r.data_source : "",
        ) for r in rows
    ]
    return record_run_rows(section, backend, threads, benchmarks; kwargs...)
end

"""
    record_run_rows(section, backend, threads, benchmarks::Vector{<:AbstractDict}; ts, source, kwargs...)

Same as [`record_run`](@ref) but `benchmarks` is already JSON-shaped (one `Dict` per row, keys
`category`/`method`/`framework`/`threads`/`time_ms`/`nrmse_gt`/`nrmse_ristretto`). Used directly by
[`record_run`](@ref) and by `migrate_to_store.jl`, which passes rows parsed back out of the old
per-section JSON files and overrides `ts`/`source` to preserve when/where they actually ran instead
of stamping them as recorded now. `root` is the store (see the module header).
"""
function record_run_rows(
        section::AbstractString, backend::AbstractString, threads::Integer, benchmarks::AbstractVector;
        ts::AbstractString = Dates.format(now(), dateformat"yyyymmdd-HHMMSS-sss"),
        source::AbstractString = source_tag(), root::AbstractString = RESULTS_DIR, kwargs...,
    )
    dir = runs_dir(root)
    mkpath(dir)
    # The SLURM job id lets a job pick out its own files from a results directory that concurrent
    # jobs on other checkouts also write to (`benchmark/slurm/gpu_refs.sh`).
    job = haskey(ENV, "SLURM_JOB_ID") ? "_job$(ENV["SLURM_JOB_ID"])" : ""
    run_id = "$(ts)_$(backend)_$(threads)threads_$(section)_$(source)$(job)_pid$(getpid())"
    path = joinpath(dir, "$run_id.json")
    ispath(path) && (path = joinpath(dir, "$(run_id)_$(rand(UInt32)).json"))
    meta = Dict{String, Any}(string(k) => v for (k, v) in kwargs)
    haskey(meta, "schema_version") || (meta["schema_version"] = SCHEMA_VERSION)
    open(path, "w") do io
        JSON.print(
            io,
            merge(
                meta,
                Dict(
                    "run_id" => run_id, "ts" => ts, "source" => source, "section" => section,
                    "backend" => backend, "threads" => threads, "benchmarks" => benchmarks,
                ),
            ),
            2,
        )
    end
    return path
end

_json_num(x::Real) = isfinite(x) ? Float64(x) : -1.0
_json_num(x) = x  # already JSON-safe (e.g. read back from a migrated file)

"""
    run_files(root = RESULTS_DIR) -> Vector{String}

Every recorded run file of the store at `root`, oldest first (filename timestamp order).
"""
function run_files(root::AbstractString = RESULTS_DIR)
    dir = runs_dir(root)
    isdir(dir) || return String[]
    return sort(filter(f -> endswith(f, ".json"), readdir(dir; join = true)))
end

"""
    load_run_files(root) -> Vector{Dict{String, Any}}

Every readable run file of the store at `root`, parsed, oldest first. Unreadable files (a run
killed mid-write) are skipped with a warning.
"""
function load_run_files(root::AbstractString)
    out = Dict{String, Any}[]
    for f in run_files(root)
        try
            push!(out, JSON.parsefile(f))
        catch e
            @warn "unreadable run file, skipping" f exception = e
        end
    end
    return out
end

"""
    Row

One benchmark row, flattened with its run's metadata (backend, threads, source, ts). Shared by
`query_results.jl` and `export_snapshot.jl` so both read `results/runs/` the same way.
"""
struct Row
    backend::String
    threads::Int
    category::String
    method::String
    framework::String
    time_ms::Float64
    nrmse_gt::Float64
    nrmse_ristretto::Float64
    source::String
    ts::String
    case_id::String
    data_source::String
    schema_version::Int
end

"""
    load_rows(root = RESULTS_DIR; schema_version = SCHEMA_VERSION) -> Vector{Row}

Every comparison row from every run file of the store at `root`, flattened. Rows from an older
schema (measured on other inputs than the catalog's cases) are left out unless `schema_version`
is lowered.
"""
function load_rows(root::AbstractString = RESULTS_DIR; schema_version::Integer = SCHEMA_VERSION)
    rows = Row[]
    for d in load_run_files(root)
        v = get(d, "schema_version", 1)
        v < schema_version && continue
        backend = get(d, "backend", "")
        threads = get(d, "threads", 0)
        source = get(d, "source", "unknown")
        ts = get(d, "ts", "")
        for b in get(d, "benchmarks", [])
            haskey(b, "category") || continue
            push!(
                rows, Row(
                    backend, threads, b["category"], b["method"], b["framework"],
                    Float64(b["time_ms"]), Float64(b["nrmse_gt"]), Float64(b["nrmse_ristretto"]), source, ts,
                    get(b, "case_id", ""), get(b, "data_source", ""), v,
                )
            )
        end
    end
    return rows
end

"""
    latest_per_case(rows; prefer_source = nothing) -> Vector{Row}

One row per (backend, threads, case, category, method, framework): the most recent (`ts`), among
rows matching `prefer_source` if given, else preferring `"slurm"` over anything else and falling
back to whatever exists when no `"slurm"` row is present for that case. An accuracy-race method
(`"… (NRMSE≤0.05, 150 it)"`) is keyed without its iteration count, which is the race's outcome, so
a rerun that reaches the target sooner replaces the earlier row.
"""
function latest_per_case(rows::Vector{Row}; prefer_source::Union{Nothing, AbstractString} = nothing)
    best = Dict{NTuple{6, Any}, Row}()
    for r in rows
        prefer_source !== nothing && r.source != prefer_source && continue
        key = (r.backend, r.threads, r.case_id, r.category, _method_key(r.method), r.framework)
        cur = get(best, key, nothing)
        if cur === nothing
            best[key] = r
        else
            better = prefer_source !== nothing ? r.ts > cur.ts :
                (r.source == "slurm", r.ts) > (cur.source == "slurm", cur.ts)
            better && (best[key] = r)
        end
    end
    return collect(values(best))
end

_method_key(method::AbstractString) = replace(method, r"(\(NRMSE≤[^,]*), \d+ it\)$" => s"\1)")

end

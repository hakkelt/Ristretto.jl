# Write the committed snapshot: `results/benchmark_<backend>_<n>threads.json`, one per (backend,
# threads) pair found in `results/runs/`, each holding the latest `source = "slurm"` row per (case,
# category, method, framework). The snapshot is the reviewable, versioned record of a full cluster
# run; `results/runs/` itself is local working data. There is no merge of raw fragments, only a
# query (`ResultsStore.latest_per_case`) written out to a stable filename.
#
# Snapshot schema 3 adds a `run` record next to the rows: when the rows were measured, on which
# hardware (CPU model and thread count, GPU), Julia version and BLAS library, gathered from the
# run files the rows come from. Runs recorded before the harness stored the CPU model take it from
# `--cpu-model` (`AMD EPYC 7763` for the cluster the committed snapshots come from).
#
#   julia --project=benchmark/comparison benchmark/comparison/scripts/export_snapshot.jl
#   julia --project=benchmark/comparison benchmark/comparison/scripts/export_snapshot.jl --backend=mkl --threads=16
#   julia --project=benchmark/comparison benchmark/comparison/scripts/export_snapshot.jl --cpu-model="AMD EPYC 7763 64-Core Processor"
using JSON

include(joinpath(@__DIR__, "..", "..", "utils", "results_store.jl"))
using .ResultsStore: load_rows, latest_per_case, load_run_files, RESULTS_DIR

const SNAPSHOT_SCHEMA_VERSION = 3

const ARG = Dict(
    m.captures[1] => m.captures[2] for m in (match(r"^--([\w-]+)=(.*)$", a) for a in ARGS) if m !== nothing
)

const RUN_FILES = load_run_files(RESULTS_DIR)
rows = load_rows(RUN_FILES)
isempty(rows) && (@info "no rows in results/runs/ -- nothing to export"; exit(0))

haskey(ARG, "backend") && filter!(r -> r.backend == ARG["backend"], rows)
haskey(ARG, "threads") && filter!(r -> r.threads == parse(Int, ARG["threads"]), rows)

# The metadata of every run file, by run id: what a row records of its run.
const RUNS = Dict(get(d, "run_id", "") => d for d in RUN_FILES)

values_of(metas, key) = sort(unique(string(m[key]) for m in metas if get(m, key, nothing) !== nothing))
date_of(ts) = "$(ts[1:4])-$(ts[5:6])-$(ts[7:8])"

# The versions of the runs recorded before the harness stored them, per backend family.
const INJECTED_VERSIONS = JSON.parsefile(joinpath(RESULTS_DIR, "toolkit_versions.json"))

# A row's toolkit is the first word of its framework label; its version is the one its run
# recorded, or for a run that recorded none, the injected one.
toolkit(r) = first(split(r.framework))
function toolkit_version(r)
    recorded = get(get(RUNS, r.run_id, Dict{String, Any}()), "toolkit_versions", nothing)
    recorded === nothing && (recorded = INJECTED_VERSIONS[r.backend == "cuda" ? "cuda" : "cpu"])
    v = get(recorded, toolkit(r), nothing)
    return v === nothing ? nothing : string(v)
end
# A Ristretto version carries the commit it was measured at; the release is what identifies it.
release(v) = replace(v, r" \(.*\)$" => "")

# A toolkit's rows measured with an older release than its latest run's are left out, also where
# the latest release recorded no row for the case (a solver that no longer reaches the accuracy
# target, say): a table would otherwise mix releases and show a time the current one cannot reach.
function current_release_rows(rows)
    newest = Dict{Tuple{String, String}, Tuple{String, String}}()
    for r in rows
        v = toolkit_version(r)
        v === nothing && continue
        key = (r.backend == "cuda" ? "cuda" : "cpu", toolkit(r))
        (!haskey(newest, key) || r.ts > newest[key][1]) && (newest[key] = (r.ts, release(v)))
    end
    return filter(rows) do r
        v = toolkit_version(r)
        key = (r.backend == "cuda" ? "cuda" : "cpu", toolkit(r))
        v === nothing || !haskey(newest, key) || release(v) == newest[key][2]
    end
end

# A run restricted to some competitors (`--frameworks=`) still times Ristretto, as the reference
# its rows are compared against; those Ristretto rows are a by-product of the competitors' rerun,
# not a measurement of Ristretto, and are left out.
function measured(r)
    patterns = get(get(RUNS, r.run_id, Dict{String, Any}()), "frameworks_filter", nothing)
    return patterns === nothing || any(p -> occursin(p, lowercase(r.framework)), patterns)
end

latest = latest_per_case(current_release_rows(filter(measured, rows)); prefer_source = "slurm")
isempty(latest) && (@info "no source=\"slurm\" rows match; nothing exported (rerun on the cluster first)"; exit(0))

# Per toolkit, the versions of the runs behind its rows.
function toolkit_versions(section_rows)
    out = Dict{String, Vector{String}}()
    for r in section_rows
        v = toolkit_version(r)
        v === nothing || union!(get!(out, toolkit(r), String[]), [v])
    end
    return Dict(k => sort(v) for (k, v) in out)
end

function run_record(section_rows)
    metas = [RUNS[r.run_id] for r in section_rows if haskey(RUNS, r.run_id)]
    dates = sort(unique(date_of(r.ts) for r in section_rows))
    cpu = values_of(metas, "cpu_model")
    isempty(cpu) && haskey(ARG, "cpu-model") && (cpu = [ARG["cpu-model"]])
    return Dict(
        "dates" => [first(dates), last(dates)],
        "cpu_model" => cpu,
        "cpu_threads" => values_of(metas, "cpu_threads"),
        "gpu" => values_of(metas, "gpu"),
        "hostnames" => values_of(metas, "hostname"),
        "julia_version" => values_of(metas, "julia_version"),
        "blas_vendor" => unique(basename.(values_of(metas, "blas_vendor"))),
        "toolkit_versions" => toolkit_versions(section_rows),
    )
end

for backend in sort(unique(r.backend for r in latest))
    for threads in sort(unique(r.threads for r in latest if r.backend == backend))
        section_rows = filter(r -> r.backend == backend && r.threads == threads, latest)
        isempty(section_rows) && continue
        path = joinpath(RESULTS_DIR, "benchmark_$(backend)_$(threads)threads.json")
        open(path, "w") do io
            JSON.print(
                io,
                Dict(
                    "backend" => backend, "threads" => threads, "schema_version" => SNAPSHOT_SCHEMA_VERSION,
                    "run" => run_record(section_rows),
                    "benchmarks" => [
                        Dict(
                            "case_id" => r.case_id, "data_source" => r.data_source,
                            "category" => r.category, "method" => r.method, "framework" => r.framework,
                            "threads" => r.threads, "time_ms" => r.time_ms,
                            "nrmse_gt" => r.nrmse_gt, "nrmse_ristretto" => r.nrmse_ristretto,
                        ) for r in sort(section_rows; by = r -> (r.case_id, r.category, r.method, r.framework))
                    ],
                ),
                4,
            )
        end
        @info "exported" path n = length(section_rows)
    end
end

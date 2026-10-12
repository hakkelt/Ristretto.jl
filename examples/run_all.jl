#!/usr/bin/env julia
#
# Run every example, or the subset selected by the command-line arguments, and print a
# pass/fail table at the end. Each example is `include`d in turn, so a failure in one does not
# stop the rest; the exception is reported and the run carries on.
#
#   julia --project=examples examples/run_all.jl                 # everything
#   julia --project=examples examples/run_all.jl m4raw ocmr       # two sources
#   julia --project=examples examples/run_all.jl fastmri/brain    # by path fragment
#
# The large files (`BIG`) are excluded unless their source or path is named explicitly: with the
# fastMRI prostate, breast and multicoil knee data and the large mridata.org files, a full run
# downloads roughly 20 GB.
#
# Set `SYNAPSE_AUTH_TOKEN` for the CMRxRecon sources and register fastMRI URLs with
# `MRITestData.set_fastmri_urls!` before including those.

using Printf

include(joinpath(@__DIR__, "ExampleUtils.jl"))

const BIG = [
    "fastmri/prostate_t2.jl", "fastmri/prostate_diffusion.jl", "fastmri/breast_stack_of_stars.jl",
    "fastmri/knee_multicoil_undersampled.jl", "fastmri/knee_multicoil_fully_sampled.jl",
    "mridata/knee_3d_fully_sampled.jl", "mridata/other_3d_undersampled.jl",
    "mridata/knee_2d_fully_sampled.jl", "mridata/brain_3d_with_calibration_block.jl",
]

scripts = String[]
for (root, _, files) in walkdir(@__DIR__), file in sort(files)
    endswith(file, ".jl") || continue
    path = relpath(joinpath(root, file), @__DIR__)
    path in ("run_all.jl", "ExampleUtils.jl") && continue
    push!(scripts, path)
end
sort!(scripts)

selected = if isempty(ARGS)
    filter(p -> !(p in BIG), scripts)
else
    filter(p -> any(arg -> occursin(arg, p), ARGS), scripts)
end

isempty(selected) && error("no example matches $(ARGS); available: $(join(scripts, ", "))")

results = Tuple{String, Symbol, String}[]
for path in selected
    println("\n", "="^90, "\n== ", path, "\n", "="^90)
    flush(stdout)
    try
        include(joinpath(@__DIR__, path))
        push!(results, (path, :ok, ""))
    catch err
        message = first(split(sprint(showerror, err), '\n'))
        @error "example failed" path exception = (err, catch_backtrace())
        push!(results, (path, :failed, first(message, 160)))
    end
    GC.gc()
end

println("\n", "="^90)
for (path, status, message) in results
    @printf("%-6s %-55s %s\n", status, path, message)
end
@printf("\n%d of %d examples ran\n", count(r -> r[2] == :ok, results), length(results))

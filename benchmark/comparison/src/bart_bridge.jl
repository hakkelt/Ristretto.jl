module BARTBridge

using BartIO

"""
    run_bart(num_outputs::Int, cmd::String, inputs...)

Wrapper around `BartIO.bart` that standardises the BART environment. `TOOLBOX_PATH` must name
the BART build; `_setup.jl` sets it from `RISTRETTO_BENCH_BART_MKL` / `RISTRETTO_BENCH_BART_OPENBLAS`,
and sets `BART_USE_FFTW_WISDOM` (`RISTRETTO_BENCH_BART_MEASURE`).
"""
function run_bart(num_outputs::Int, cmd::String, inputs...)
    _check_toolbox_path()
    return bart(num_outputs, cmd, inputs...)
end

"""
    run_bart_timed(num_outputs::Int, cmd::String, inputs...; runs = 1, before_each = () -> nothing) -> (outputs, seconds)

`run_bart` for a tool that reports its own run time, `pics`, run `runs` times as fresh processes
on inputs written once, with `before_each()` called before each. `seconds` holds each run's
`Total Time` as the tool prints it, measured inside the process from the start of the tool to its
outputs written and its inputs unmapped. It leaves out what the process pays before the tool starts
(loading the binary and its libraries) and the harness' writing of the inputs and reading of the
outputs, so it needs no estimate of either subtracted. The outputs, of the last run, are a vector,
as `bart` returns them for several outputs, or the array itself for one.
"""
function run_bart_timed(
        num_outputs::Int, cmd::String, inputs::Array{ComplexF32}...;
        runs::Int = 1, before_each = () -> nothing,
    )
    _check_toolbox_path()
    dir = mktempdir(; prefix = "jl_", cleanup = true)
    try
        infiles = [joinpath(dir, "in$i") for i in eachindex(inputs)]
        outfiles = [joinpath(dir, "out$i") for i in 1:num_outputs]
        foreach(write_cfl, infiles, inputs)
        command = `$(get_bart_path()) $(split(cmd)) $infiles $outfiles`
        seconds = map(1:runs) do _
            before_each()
            log = IOBuffer()
            proc = run(pipeline(ignorestatus(command); stdout = log, stderr = log))
            text = String(take!(log))
            success(proc) || error("bart $cmd failed:\n$text")
            m = match(r"Total Time: ([0-9.eE+-]+)", text)
            m === nothing && error("bart $cmd printed no `Total Time` (debug level below info?):\n$text")
            parse(Float64, m.captures[1])
        end
        outputs = [read_cfl(f) for f in outfiles]
        return (num_outputs == 1 ? only(outputs) : outputs), seconds
    finally
        rm(dir; force = true, recursive = true)
    end
end

_check_toolbox_path() = haskey(ENV, "TOOLBOX_PATH") || error(
    "TOOLBOX_PATH is not set: configure RISTRETTO_BENCH_BART_MKL / RISTRETTO_BENCH_BART_OPENBLAS in " *
        "benchmark/slurm/site.env"
)

export run_bart, run_bart_timed

end

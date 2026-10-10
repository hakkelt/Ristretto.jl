# Writes the case the startup benchmark solves to plain files, so every toolkit's script reads the
# same data without loading the comparison harness (which would load all six toolkits at once):
#
#   julia --project=benchmark/comparison benchmark/startup/prepare.jl <datadir> [case id]
#
# `<datadir>/`: `kspace.c64` (zero-filled `(nx, ny, coil)`), `smaps.c64` (`(nx, ny, coil)`),
# `reference.c64` (`(nx, ny)`), `mask.u8` (`(nx, ny)`), all column-major complex64 / uint8, the same
# k-space and maps as BART's `ksp.cfl` / `sens.cfl` (`(nx, ny, 1, coil)`), and `meta.json` with the
# sizes, the iteration counts and each toolkit's calibrated λ of the L1-wavelet row.

using JSON

include(joinpath(@__DIR__, "..", "utils", "bench_utils.jl"))
using .BenchUtils

const DIR = ARGS[1]
const CASE = length(ARGS) >= 2 ? ARGS[2] : "shepp_logan_2d_8ch_cartesian"
mkpath(DIR)

c = get_case(CASE)
c.family === :single_slice && c.trajectory === :cartesian && c.smaps !== nothing ||
    error("$CASE: the startup benchmark takes a single-slice multichannel Cartesian case")
nx, ny, nc = size(c.kspace)

writeraw(name, a) = open(io -> write(io, a), joinpath(DIR, name), "w")
writeraw("kspace.c64", ComplexF32.(c.kspace))
writeraw("smaps.c64", ComplexF32.(c.smaps))
writeraw("reference.c64", ComplexF32.(c.reference))
writeraw("mask.u8", UInt8.(c.mask))

# BART's `.cfl` is the same column-major complex64 data, its `.hdr` the dimensions.
function writecfl(name, a, dims)
    writeraw(name * ".cfl", ComplexF32.(a))
    open(joinpath(DIR, name * ".hdr"), "w") do io
        println(io, "# Dimensions")
        println(io, join(dims, " "))
    end
    return nothing
end
writecfl("ksp", c.kspace, (nx, ny, 1, nc, ntuple(_ -> 1, 12)...))
writecfl("sens", c.smaps, (nx, ny, 1, nc, ntuple(_ -> 1, 12)...))

lambda = JSON.parsefile(joinpath(@__DIR__, "..", "comparison", "results", "lambda", "$CASE.json"))["lambda"]["wavelet"]
meta = Dict(
    "case" => CASE, "nx" => nx, "ny" => ny, "coils" => nc,
    "wavelet_iterations" => OUTER_ITERATIONS, "cgsense_iterations" => CG_ITERATIONS,
    "wavelet_levels" => BenchUtils.WAVELET_LEVELS, "wavelet_name" => "db2", "lambda_wavelet" => lambda,
)
open(io -> JSON.print(io, meta, 2), joinpath(DIR, "meta.json"), "w")
println("wrote $CASE ($nx×$ny, $nc coils) to $DIR")

# Shared by the Julia toolkits' solve scripts (`solve_<toolkit>.jl`). Base only, so that including
# it loads nothing the toolkit under test would not load itself.
#
#   julia --project=benchmark/comparison -t N solve_<toolkit>.jl <datadir> <method> <nwarm> <nx> <ny> <nc> <maxit> <lambda>
#
# A script times its `using`, then the preparation of its inputs from the files `prepare.jl` wrote,
# the first solve (compilation included) and `nwarm` further solves, and prints one line
# `RESULT <import s> <setup s> <first s> <nrmse> <warm s,...>` for `run.jl`.

const DATADIR = ARGS[1]
const METHOD = Symbol(ARGS[2])
const NWARM = parse(Int, ARGS[3])
const NX, NY, NC = parse.(Int, ARGS[4:6])
const MAXIT = parse(Int, ARGS[7])
const LAMBDA = parse(Float64, ARGS[8])
const NTHREADS = Threads.nthreads()
# The L1-wavelet rows' `db2` depth, `BenchUtils.WAVELET_LEVELS`; `run.jl` checks the two agree.
const WAVELET_LEVELS = 3

readraw(T, name, dims...) = read!(joinpath(DATADIR, name), Array{T}(undef, dims...))

# The case's arrays: zero-filled k-space `(nx, ny, coil)`, maps `(nx, ny, coil)`, the ground truth
# and the sampling mask `(nx, ny)`.
load_case() = (
    kspace = readraw(ComplexF32, "kspace.c64", NX, NY, NC),
    smaps = readraw(ComplexF32, "smaps.c64", NX, NY, NC),
    reference = readraw(ComplexF32, "reference.c64", NX, NY),
    mask = readraw(UInt8, "mask.u8", NX, NY) .!= 0,
)

# The comparison harness' score: magnitudes, the estimate rescaled to the reference's norm.
function mag_nrmse(est, ref)
    a, r = abs.(est), abs.(ref)
    a .*= sqrt(sum(abs2, r) / sum(abs2, a))
    return sqrt(sum(abs2, a .- r) / sum(abs2, r))
end

"""
    measure(solve, t_import, t_setup, reference)

Time `solve()` once and `NWARM` more times, and print the `RESULT` line.
"""
function measure(solve, t_import, t_setup, reference)
    t_first = @elapsed x = solve()
    e = mag_nrmse(x, reference)
    warm = [@elapsed(solve()) for _ in 1:NWARM]
    println("RESULT ", t_import, " ", t_setup, " ", t_first, " ", e, " ", isempty(warm) ? "-" : join(warm, ","))
    return nothing
end

# MIRT's row of the startup benchmark (CG-SENSE only: MIRT ships no wavelet or TV prox): `Asense`
# built inside the timed region and `ncg` on `½‖Ax - y‖²`, as `mirt_recon` runs it in the comparison
# harness (benchmark/comparison/scripts/_toolkits.jl), FFTW at the thread count under test (at most 8).
include(joinpath(@__DIR__, "child_common.jl"))
const T_IMPORT = @elapsed @eval using MIRT
using FFTW: FFTW

t_setup = @elapsed begin
    FFTW.set_num_threads(min(NTHREADS, 8))
    METHOD === :cgsense || error("MIRT has no $METHOD")
    case = load_case()
    samp = dropdims(any(!iszero, case.kspace; dims = 3); dims = 3)
    y = reduce(hcat, [case.kspace[:, :, c][samp] for c in 1:NC])
end

function solve()
    A = MIRT.Asense(samp, case.smaps)
    x0 = zeros(ComplexF32, A._idim)
    return Array{ComplexF64}(first(MIRT.ncg([A], [v -> v - y], [v -> 1.0f0], x0; niter = MAXIT)))
end
measure(solve, T_IMPORT, t_setup, case.reference)

# MRIReco's row of the startup benchmark: the `reconstruction` call `mrireco` makes in the comparison
# harness (benchmark/comparison/scripts/_toolkits.jl), with the same parameters: unweighted data
# term, zero tolerances, and for L1-wavelet FISTA's step `0.95 / power_iterations(AHA)` estimated
# inside the timed region. BLAS stays at the thread count MRIReco chooses for itself (`with_mrireco_blas`);
# FFTW runs at the thread count under test.
include(joinpath(@__DIR__, "child_common.jl"))
const T_IMPORT = @elapsed @eval using MRIReco
using MRIReco: AcquisitionData, L1Regularization, L2Regularization
using FFTW: FFTW

const RLS = MRIReco.RegularizedLeastSquares
mkacq(k) = AcquisitionData(reshape(k, NX, NY, 1, NC, 1, 1); enc2D = true)

t_setup = @elapsed begin
    FFTW.set_num_threads(NTHREADS)
    case = load_case()
    senseMaps = reshape(case.smaps, NX, NY, 1, NC)
    reg, solver = METHOD === :wavelet ? (L1Regularization(LAMBDA), MRIReco.FISTA) :
        METHOD === :cgsense ? (L2Regularization(0.0), MRIReco.CGNR) : error("no $METHOD")
    # The normal operator `reconstruction` builds internally, for the FISTA step estimate only; built
    # outside the timed region, as `_mrireco_normal_operator` is.
    AHA = if METHOD === :wavelet
        a = mkacq(case.kspace)
        E = MRIReco.encodingOps_parallel(a, (NX, NY), senseMaps; slice = 1)
        W = MRIReco.WeightingOp(ComplexF32; weights = fill(ComplexF32(1 / sqrt(NX * NY)), size(a.kdata[1], 1)), rep = NC)
        MRIReco.normalOperator(∘(W, E[1]))
    end
end

function solve()
    rp = Dict{Symbol, Any}(
        :reco => "multiCoil", :reconSize => (NX, NY), :senseMaps => senseMaps,
        :solver => solver, :reg => reg, :iterations => MAXIT,
        :rho => AHA === nothing ? 5.0e-2 : 0.95 / RLS.power_iterations(AHA),
        :vary_rho => :none, :iterationsCG => 10,
        :absTol => 0.0, :relTol => 0.0, :tolInner => 0.0, :densityWeighting => false,
    )
    METHOD === :wavelet && (rp[:sparseTrafo] = "Wavelet")
    img = MRIReco.reconstruction(mkacq(case.kspace), rp)
    return reshape(Array{ComplexF64}(img), NX, NY)
end
measure(solve, T_IMPORT, t_setup, case.reference)

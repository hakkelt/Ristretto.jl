# How Ristretto reconstructs each catalog method. Shared by the harness and by the comparison suite's Ristretto
# rows, so the two time exactly the same call.

"""
    OUTER_ITERATIONS, CG_ITERATIONS, ADMM_RHO

The fixed effort every iterative method runs at: `OUTER_ITERATIONS` ADMM (or FISTA) iterations,
`CG_ITERATIONS` inner CG per ADMM iteration, a fixed ADMM penalty `ADMM_RHO`, and no early stop.
CG-SENSE runs `CG_ITERATIONS` iterations. Overridable with `CMP_OUTER` / `CMP_CG_ITERS`.

`ADMM_RHO` is relative to `‖𝒜‖²` on a checkout whose `reconstruct` scales a given ADMM penalty
by it (`_scale_admm_penalty`), and absolute on one that does not. The difference is large only
for radial cases (`‖𝒜‖² ≈ 2·10⁶`): there an absolute `0.05` never lets the regularizer act, and
their NRMSE from such a checkout is that of an unregularized solve, whatever `λ` says.
"""
const OUTER_ITERATIONS = parse(Int, get(ENV, "CMP_OUTER", "20"))
const CG_ITERATIONS = parse(Int, get(ENV, "CMP_CG_ITERS", "10"))
const ADMM_RHO = 5.0e-2

"""
    RADIAL_ADMM_RHO, RADIAL_LAMBDA

The ADMM penalty and λ per method for radial cases, where the Cartesian values do not carry over.
Calibrated on `shepp_logan_2d_8ch_radial` (tv, tgv, wavelet) and `torso_cine_8ch_radial` (lowrank,
llr, ttv) by NRMSE after `OUTER_ITERATIONS` iterations, over a λ grid `10^(-4:0.5:0.5)` and ρ from
0.002 to 20 relative to `‖𝒜‖²`.

NRMSE falls steadily as ρ decreases down to about 0.005. Below that the cine methods flatten
(ttv is best at 0.01, within 5% at 0.002) while the 2D ones keep improving, so `0.002` is within 5%
of each method's best and at least matches Ristretto's default adaptive penalty on every method. At 0.002 the NRMSE is tv 0.035, tgv 0.034, llr 0.087, lowrank 0.107 and ttv 0.067.
L1-wavelet runs FISTA, so only its λ was calibrated (NRMSE 0.279).

The calibration ran under `BartScaling`. The values are in `QuantileScaling` units, converted as
`λ · s_bart / s_quantile` on the calibration case (0.8435 for the 2D case, 0.7890 for the cine),
which leaves the effective regularization weight unchanged.
"""
const RADIAL_ADMM_RHO = 2.0e-3
const RADIAL_LAMBDA = Dict(
    :tv => 8.43e-4, :atv => 8.43e-4, :wavelet => 2.53e-3, :tgv => 8.43e-4, :lowrank => 2.37e-2, :llr => 2.37e-3,
    :ttv => 7.89e-4, :epr => 8.43e-4,
)

"""
    DEFAULT_LAMBDA

λ per method for Cartesian cases when no calibrated value is asked for; radial cases use
`RADIAL_LAMBDA`. The harness always uses these (through [`default_lambda`](@ref)), so a timing
and its NRMSE are comparable across checkouts regardless of later recalibration.

They were set under `BartScaling` (tv 0.01, wavelet 0.005, tgv 0.003, lowrank 0.01) and are now in
`QuantileScaling` units: the spatial penalties are multiplied by 1.636, the geometric mean of
`s_bart / s_quantile` over the Cartesian 2D, 3D and multi-slice cases (1.25–1.97), and the temporal
ones by 0.669, its value on the Cartesian cine. The edge-preserving roughness penalty (`:epr`, at
its default `δ`) takes anisotropic TV's λ, the penalty it approaches as `δ → 0`; it is uncalibrated.
"""
const DEFAULT_LAMBDA = Dict(
    :tv => 0.0164, :atv => 0.0164, :wavelet => 0.0082, :tgv => 0.0049, :lowrank => 0.00669, :llr => 0.00669,
    :ttv => 0.00669, :epr => 0.0164,
)

"""
    default_lambda(c::BenchCase, method) -> λ

`RADIAL_LAMBDA` for a radial case and `DEFAULT_LAMBDA` otherwise; `0.0` for an unregularized
method.
"""
default_lambda(c::BenchCase, method::Symbol) =
    get(c.trajectory === :noncartesian ? RADIAL_LAMBDA : DEFAULT_LAMBDA, penalty_of(method), 0.0)

"""
    admm_rho(c::BenchCase) -> ρ

`RADIAL_ADMM_RHO` for a radial case and `ADMM_RHO` otherwise.
"""
admm_rho(c::BenchCase) = c.trajectory === :noncartesian ? RADIAL_ADMM_RHO : ADMM_RHO

"""
    WAVELET_LEVELS

Decomposition depth of the L1-wavelet rows (`db2`), matched across toolkits.
"""
const WAVELET_LEVELS = parse(Int, get(ENV, "CMP_WAVELET_LEVELS", "3"))

"""
    ristretto_regularizer(c::BenchCase, method, λ)

`:tv` is the isotropic `λ Σ ‖∇x‖₂` (per voxel), `:atv` the anisotropic `λ Σᵢ ‖Δⁱx‖₁`. They are
different problems with different optima: on `shepp_logan_3d_8ch_cartesian` the isotropic one
reaches NRMSE 0.008 where the anisotropic one reaches 0.0033.
"""
function ristretto_regularizer(c::BenchCase, method::Symbol, λ::Real)
    method = penalty_of(method)
    vol = c.family === :volume
    method === :tv && return vol ? TotalVariation3D(λ) : TotalVariation2D(λ)
    method === :atv && return vol ? AnisotropicTotalVariation3D(λ) : AnisotropicTotalVariation2D(λ)
    method === :wavelet && return vol ?
        L1Wavelet3D(λ; wavelet = Ristretto.WT.db2, levels = WAVELET_LEVELS) :
        L1Wavelet2D(λ; wavelet = Ristretto.WT.db2, levels = WAVELET_LEVELS)
    method === :epr && return vol ? EdgePreservingRoughness3D(λ) : EdgePreservingRoughness2D(λ)
    method === :tgv && return TotalGeneralizedVariation2D(λ; ratio = 2.0)
    method === :lowrank && return LowRank(λ; time_dim = :time)
    method === :llr && return LocallyLowRank(λ; block_size = (8, 8), time_dim = :time)
    method === :ttv && return TemporalTotalVariation(λ; time_dim = :time)
    throw(ArgumentError("$method has no regularizer"))
end

"""
    PDHG_ITERATIONS

The iteration count of a PDHG row: `OUTER_ITERATIONS × CG_ITERATIONS`, the normal-operator
applications of the matching ADMM row. A PDHG iteration applies the operator about once where an
ADMM iteration applies it once per inner CG step, so equal iteration counts would not be equal
effort.
"""
const PDHG_ITERATIONS = OUTER_ITERATIONS * CG_ITERATIONS

"""
    default_maxit(method) -> Int

`CG_ITERATIONS` for CG-SENSE, `PDHG_ITERATIONS` for a PDHG or L-BFGS row (each iteration
applies the normal operator about once), `OUTER_ITERATIONS` otherwise.
"""
default_maxit(m::Symbol) =
    m === :cgsense ? CG_ITERATIONS : (haskey(PDHG_METHODS, m) || m === :epr_lbfgs) ? PDHG_ITERATIONS : OUTER_ITERATIONS

"""
    ristretto_algorithm(method, maxit; rho = ADMM_RHO)

Fixed-ρ ADMM with a fixed inner CG and no early stop for every regularized method except
L1-wavelet, which runs FISTA (forcing a fixed-ρ ADMM on it wrecks it), or POGM for
`:wavelet_pogm`; L-BFGS for `:epr_lbfgs`; CGNR for CG-SENSE;
`ChambollePock` for a PDHG row, with the step sizes `reconstruct` derives (its block-diagonal,
density-compensated preconditioning, ahead of Vũ-Condat on both Cartesian and radial data).
"""
function ristretto_algorithm(method::Symbol, maxit::Int; rho::Real = ADMM_RHO)
    method === :cgsense && return Ristretto.CGNR(; maxit, tol = 0.0)
    method === :wavelet && return Ristretto.FISTA(; maxit, tol = 0.0)
    method === :wavelet_pogm && return Ristretto.POGM(; maxit, tol = 0.0)
    method === :epr_lbfgs && return Ristretto.LBFGS(; maxit, tol = 0.0)
    haskey(PDHG_METHODS, method) && return Ristretto.ChambollePock(; maxit, tol = 0.0)
    return Ristretto.ADMM(; rho, maxit, tol = 0.0, cg_tol = 0.0, cg_maxit = CG_ITERATIONS)
end

"""
    ristretto_reconstructor(c, method; λ = default_lambda(c, method), rho = admm_rho(c), maxit, acq = nothing, device = nothing) -> () -> image

A zero-argument closure running Ristretto's reconstruction of `c` by `method`, for `time_run`, with the
ADMM penalty `rho` (relative to `‖𝒜‖²`; ignored by the methods that do not run ADMM). The
acquisition is built once, outside the closure; `acq` passes one in (the comparison suite reuses
it across rows). `maxit` defaults to [`default_maxit`](@ref).
`maxit` and `reltol = 0` are set on `IterativeReconstruction` as well as on the algorithm: the
method's values win over the algorithm's, so both must agree to run the full count.

`device`, a GPU array type such as `CuArray`, reconstructs on that device: the closure moves the
host acquisition there with `adapt` and copies the image back into a host `Array`, so the time is
from host data to host image, as it is for a toolkit that takes and returns host arrays.
"""
function ristretto_reconstructor(
        c::BenchCase, method::Symbol;
        λ::Real = default_lambda(c, method),
        rho::Real = admm_rho(c),
        maxit::Int = default_maxit(method),
        acq = nothing,
        device = nothing,
    )
    direct = method === :adjoint || method === :gridding
    a = something(acq, ristretto_acquisition(c; dcf = method === :gridding))
    m = if direct
        DirectReconstruction()
    else
        reg = method === :cgsense ? () : ristretto_regularizer(c, method, λ)
        IterativeReconstruction(; regularization = reg, algorithm = ristretto_algorithm(method, maxit; rho), maxit, reltol = 0.0)
    end
    device === nothing && return () -> reconstruct(a, m; verbosity = Silent())
    return () -> Array(reconstruct(Ristretto.Adapt.adapt(device, a), m; verbosity = Silent()))
end

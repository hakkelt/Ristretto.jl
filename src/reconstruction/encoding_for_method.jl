"""
    TemporalBasis(Φ::AbstractMatrix; time_dim = nothing)

Signal model representing dynamic image series expanded in a temporal subspace:
``x(r, t) = \\sum_{k=1}^K \\Phi(t, k) c(r, k)``.

# Arguments
- `Φ`: ``N_t \\times K`` basis matrix where ``N_t`` is the number of time frames and ``K`` is the number of subspace coefficients.
- `time_dim`: (optional) Dimension index or name for the temporal dimension (defaults to `:time` or last image dimension).
"""
struct TemporalBasis{T, M <: AbstractMatrix{T}, D}
    Φ::M
    time_dim::D
    function TemporalBasis(Φ::M; time_dim::D = nothing) where {T, M <: AbstractMatrix{T}, D}
        _check_dim_spec(time_dim, "time_dim")
        return new{T, M, D}(Φ, time_dim)
    end
end

"""
    signal_model_operator(method::ReconstructionMethod, acq::AcquisitionInfo; threaded::Bool)

Constructs the signal model linear operator `ℳ` for the given reconstruction method and acquisition data.
Returns `nothing` if no signal model is specified.
"""
signal_model_operator(::ReconstructionMethod, ::AcquisitionInfo; threaded::Bool = true) = nothing

function signal_model_operator(method::IterativeReconstruction, acq::AcquisitionInfo; threaded::Bool = true)
    return signal_model_operator(method.signal_model, acq; threaded)
end

signal_model_operator(::Nothing, ::AcquisitionInfo; threaded::Bool = true) = nothing
signal_model_operator(::KSpaceToImage, ::AcquisitionInfo; threaded::Bool = true) = nothing

function signal_model_operator(model::TemporalBasis, acq::AcquisitionInfo; threaded::Bool = true)
    img_dims = get_image_dims(acq)
    img_size = get_image_size(acq)
    time_dim_idx = get_time_dim(model.time_dim, img_dims)
    Nt = size(model.Φ, 1)
    K = size(model.Φ, 2)
    @argcheck img_size[time_dim_idx] == Nt "TemporalBasis time frame count ($Nt) does not match acquisition time dimension size ($(img_size[time_dim_idx]))"

    coeff_size = ntuple(i -> i == time_dim_idx ? K : img_size[i], length(img_size))
    T = complex(eltype(model.Φ))
    at = _array_type_of(acq)
    Φᵀ = _to_storage_of(acq, Matrix(transpose(model.Φ)))

    # Construct matrix multiplication operator along time_dim
    # For trailing time_dim, flat spatial dimension is prod(img_size[1:end-1])
    if time_dim_idx == length(img_size)
        N_spatial = prod(img_size[1:(end - 1)])
        R_in = Reshape(Eye(T, (N_spatial, K); array_type = at), coeff_size...)
        L = LMatrixOp(T, (N_spatial, K), Φᵀ; threaded)
        R_out = Reshape(Eye(T, (N_spatial, Nt); array_type = at), img_size...)
        ℳ = R_out * (L * R_in')
    else
        # General case via PermuteDims to trailing axis, LMatrixOp, and permute back.
        # `perm` moves the time/coeff axis to the last position (output dim time_dim_idx..N-1
        # come from input time_dim_idx+1..N, output dim N from input time_dim_idx); `inv_perm`
        # is its inverse and restores the original axis order.
        N = length(img_size)
        perm = ntuple(i -> i < time_dim_idx ? i : (i == N ? time_dim_idx : i + 1), N)
        inv_perm = ntuple(i -> i == time_dim_idx ? N : (i >= time_dim_idx ? i - 1 : i), N)
        perm_img_size = ntuple(i -> img_size[perm[i]], length(img_size))
        perm_coeff_size = ntuple(i -> coeff_size[perm[i]], length(coeff_size))

        N_spatial = prod(perm_img_size[1:(end - 1)])
        P_in = PermuteDims(T, coeff_size, perm; array_type = at)
        R_in = Reshape(Eye(T, (N_spatial, K); array_type = at), perm_coeff_size...)
        L = LMatrixOp(T, (N_spatial, K), Φᵀ; threaded)
        R_out = Reshape(Eye(T, (N_spatial, Nt); array_type = at), perm_img_size...)
        P_out = PermuteDims(T, perm_img_size, inv_perm; array_type = at)
        ℳ = P_out * R_out * L * R_in' * P_in
    end

    if _has_dimnames(acq.kspace_data)
        in_dimnames = ntuple(i -> i == time_dim_idx ? :coeff : img_dims[i], length(img_dims))
        out_dimnames = img_dims
        ℳ = NamedDimsOp{in_dimnames, out_dimnames}(ℳ)
    end
    return ℳ
end

# Dimension queries for signal models and methods

get_affected_dims(::Nothing, ::Any, ::Any) = ()
get_affected_dims(model::TemporalBasis, ::Any, image_dims) = (image_dims[get_time_dim(model.time_dim, image_dims)],)
get_affected_dims(::KSpaceToImage, ::Any, image_dims) = image_dims

variable_dims(method::IterativeReconstruction, acq::AcquisitionInfo) = variable_dims(method.signal_model, acq)
variable_dims(::Nothing, acq::AcquisitionInfo) = get_image_dims(acq)
function variable_dims(model::TemporalBasis, acq::AcquisitionInfo)
    img_dims = get_image_dims(acq)
    t_idx = get_time_dim(model.time_dim, img_dims)
    return ntuple(i -> i == t_idx ? :coeff : img_dims[i], length(img_dims))
end
variable_dims(::KSpaceToImage, acq::AcquisitionInfo) = dimnames(acq.kspace_data)

variable_size(method::IterativeReconstruction, acq::AcquisitionInfo) = variable_size(method.signal_model, acq)
variable_size(::Nothing, acq::AcquisitionInfo) = get_image_size(acq)
function variable_size(model::TemporalBasis, acq::AcquisitionInfo)
    img_size = get_image_size(acq)
    t_idx = get_time_dim(model.time_dim, get_image_dims(acq))
    return ntuple(i -> i == t_idx ? size(model.Φ, 2) : img_size[i], length(img_size))
end
function variable_size(::KSpaceToImage, acq::AcquisitionInfo)
    # `KSpaceToImage` optimizes over a *full, dense* k-space grid, which the partitioned layout is
    # precisely not.
    _reject_partitioned(acq.kspace_data, "the KSpaceToImage signal model")
    img_sz = get_image_size(acq)
    return (img_sz[1], img_sz[2], size(acq.kspace_data)[3:end]...)
end

output_dims(method::IterativeReconstruction, acq::AcquisitionInfo) = output_dims(method.signal_model, acq)
output_dims(::Nothing, acq::AcquisitionInfo) = get_image_dims(acq)
output_dims(::TemporalBasis, acq::AcquisitionInfo) = get_image_dims(acq)
function output_dims(model::KSpaceToImage, acq::AcquisitionInfo)
    return model.coil_combination isa NoCoilCombination ? get_image_dims(acq) : filter(!=(:coil), get_image_dims(acq))
end

"""
    model_encoding_operator(model, acq::AcquisitionInfo; threaded::Bool, fast_planning::Bool)

Encoding operator mapping the reconstruction optimization variable to the measured k-space:
- `nothing` — the physical encoding operator `𝒜`.
- `TemporalBasis` — `𝒜 * ℳ`, with `ℳ` expanding subspace coefficients to the image series.
- `KSpaceToImage` — just the subsampling operator `𝒫` (the variable *is* k-space).
"""
function model_encoding_operator(::Nothing, acq::AcquisitionInfo; threaded::Bool, fast_planning::Bool)
    return get_encoding_operator(acq; threaded, fast_planning)
end

function model_encoding_operator(model::TemporalBasis, acq::AcquisitionInfo; threaded::Bool, fast_planning::Bool)
    𝒜 = get_encoding_operator(acq; threaded, fast_planning)
    ℳ = signal_model_operator(model, acq; threaded)
    if 𝒜 isa NamedDimsOp && ℳ isa NamedDimsOp
        @argcheck dimnames(𝒜, 2) == dimnames(ℳ, 1) "signal model codomain does not match encoding operator domain"
    end
    return 𝒜 * ℳ
end

function model_encoding_operator(::KSpaceToImage, acq::AcquisitionInfo; threaded::Bool, fast_planning::Bool)
    isnothing(acq.subsampling) || return get_subsampling_operator(acq; threaded)
    raw = unname(acq.kspace_data)
    P = Eye(eltype(raw), size(raw)...; array_type = _array_type_of(raw))
    return _has_dimnames(acq.kspace_data) ?
        NamedDimsOp{dimnames(acq.kspace_data), dimnames(acq.kspace_data)}(P) : P
end

"""
    apply_signal_model(model, x̂, acq::AcquisitionInfo; threaded::Bool)

Map a solved optimization variable `x̂` to the output image: identity for `nothing`, `ℳ * x̂` for
`TemporalBasis`, and an inverse Fourier transform + coil combination for `KSpaceToImage`.
"""
apply_signal_model(::Nothing, x̂, ::AcquisitionInfo; threaded::Bool) = x̂
function apply_signal_model(model::TemporalBasis, x̂, acq::AcquisitionInfo; threaded::Bool)
    return signal_model_operator(model, acq; threaded) * x̂
end
function apply_signal_model(model::KSpaceToImage, x̂, acq::AcquisitionInfo; threaded::Bool)
    return _kspace_to_image(x̂, model.coil_combination, acq.sensitivity_maps, acq)
end

"""
    build_encoding_operator(acq::AcquisitionInfo, method::ReconstructionMethod; threaded::Bool = true, fast_planning::Bool = false)

Builds the encoding operator mapping the reconstruction optimization variable to the measured
k-space, dispatching on the method's signal model (via the internal `model_encoding_operator`).
"""
function build_encoding_operator(
        acq::AcquisitionInfo,
        method::ReconstructionMethod;
        threaded::Bool = true,
        fast_planning::Bool = false,
    )
    model = method isa IterativeReconstruction ? method.signal_model : nothing
    return model_encoding_operator(model, acq; threaded, fast_planning)
end

"""
    _fast_planning(method, acq, config; threaded = config.threaded) -> Bool

Whether `method`'s encoding operator plans its FFTs with `FFTW.ESTIMATE` (`true`) rather than
`FFTW.MEASURE`, as `config.fft_planning` asks: `:estimate` and `:measure` force the choice,
`:auto` takes `MEASURE` when [`_measure_score`](@ref) is not negative. Under a provider other
than FFTW itself (MKL) the planner flags do nothing, and `ESTIMATE` is returned.
"""
function _fast_planning(method::ReconstructionMethod, acq::AcquisitionInfo, config; threaded::Bool = config.threaded)
    config.fft_planning === :auto || return config.fft_planning === :estimate
    FFTW.fftw_provider == "fftw" || return true
    return _measure_score(method, acq; threaded) < 0
end

"""
    _measure_score(method, acq; threaded) -> Float64

Whether planning the encoding's FFTs with `FFTW.MEASURE` pays for itself, as a sum of `log2`
scores; `MEASURE` is chosen when the sum is not negative. `MEASURE` times candidate algorithms
on the real arrays, which costs about 0.1–0.2 s per 2D transform and 1–1.5 s per 3D one, whatever
the batch, and pays back at every later transform by running faster than the plan `ESTIMATE`
guesses. It pays when

    transforms × batch × gain(grid, threads) ≥ plans × cost(grid)

and the score is that inequality taken in `log2`, one term per factor:

- `log2` of the transforms the reconstruction runs per batch member ([`_fft_transforms`](@ref):
  iterations × operator applications per iteration × 2 for an iterative method, 1 for a direct
  one);
- `log2` of the batch, every axis after the sample axes (coils, frames, …);
- the grid's score, `log2(gain / cost)` for one transform of the grid on one thread, which rises
  steeply with the grid: `ESTIMATE` is within 2× of `MEASURE` below about 2¹⁴ points, and 7–9×
  slower from 2¹⁶ on, a 256² grid (the 2× oversampled grid of a 128² non-Cartesian image)
  included;
- the thread score: every doubling of FFTW's threads shrinks the gap between the two plans;
- a constant for the transforms planned per operator (forward, adjoint, and the normal
  operator's pair).

The grid and thread scores were fitted to single FFT timings, and checked against whole
reconstructions planned from scratch (1, 4 and 16 threads, 2D, radial, 3D and cine, CG-SENSE,
ADMM and PDHG): of those 36, the score picks the faster planner in all 35 that are not a tie; see "Performance & Threading" in
the manual.
"""
function _measure_score(method::ReconstructionMethod, acq::AcquisitionInfo; threaded::Bool)
    points, batch = _fft_grid_and_batch(acq)
    nthreads = threaded ? Threads.nthreads() : 1
    return log2(_fft_transforms(method)) + log2(batch) + _grid_score(points) -
        MEASURE_THREAD_SCORE * log2(nthreads) - MEASURE_PLANS_SCORE
end

# Per doubling of FFTW's threads; see `_measure_score`.
const MEASURE_THREAD_SCORE = 1.25
# `log2` of the transforms one operator plans, less the one for the two transforms an operator
# application runs.
const MEASURE_PLANS_SCORE = 1.0

# `log2(gain / cost)` of one transform of a `points`-point grid on one thread: the time `MEASURE`'s
# plan saves over `ESTIMATE`'s per transform, over what planning it costs. -12 up to 2¹⁴ points,
# -7 at 2¹⁶, linear in `log2(points)` between, and rising slowly beyond (-6.25 at a 128³ grid).
_grid_score(points::Real) = max(-12.0, min(-12 + 2.5 * (log2(points) - 14), -7 + 0.15 * (log2(points) - 16)))

# The transform size and the batch of the encoding's FFTs: the image grid (oversampled twice per
# axis for a non-uniform transform, as its Toeplitz normal operator is) and the product of every
# axis after the sample axes.
function _fft_grid_and_batch(acq::AcquisitionInfo)
    grid = prod(acq.image_size)
    acq isa NonCartesianAcquisitionInfo && (grid *= 2^length(acq.image_size))
    batch = prod(_ksp_trailing_size(acq.kspace_data, _get_sample_dims_count(acq) + 1); init = 1)
    return grid, batch
end

"""
    _fft_transforms(method) -> Int

How many times a reconstruction with `method` transforms each batch member: once for a direct
reconstruction, and for an iterative one twice (forward and adjoint) per operator application,
times the applications per iteration ([`_applications_per_iteration`](@ref)), times the
iterations, `method.maxit` or else the algorithm's own.
"""
_fft_transforms(::ReconstructionMethod) = 1
function _fft_transforms(method::IterativeReconstruction)
    maxit = something(method.maxit, _algorithm_maxit(method.algorithm))
    return 2 * max(1, maxit) * _applications_per_iteration(method.algorithm)
end

_algorithm_maxit(alg::ProximalAlgorithms.IterativeAlgorithm) = alg.maxit
_algorithm_maxit(algs::Tuple) = minimum(_algorithm_maxit, algs)
_algorithm_maxit(_) = 100

"""
    _applications_per_iteration(algorithm) -> Int

Operator applications (forward and adjoint pairs) per iteration of `algorithm`: one for CG, the
forward-backward family and the primal-dual family, and one per inner CG step plus the right-hand
side for ADMM, whose inner CG is counted up to `ADMM_COUNTED_CG_STEPS` steps, since it stops at
its tolerance long before a large `cg_maxit`. A tuple of candidates, resolved only once the model
is built, counts as its cheapest member.
"""
_applications_per_iteration(_) = 1
_applications_per_iteration(algs::Tuple) = minimum(_applications_per_iteration, algs)
function _applications_per_iteration(alg::ProximalAlgorithms.IterativeAlgorithm{<:ProximalAlgorithms.ADMMIteration})
    return min(get(alg.kwargs, :cg_maxit, ADMM_COUNTED_CG_STEPS), ADMM_COUNTED_CG_STEPS) + 1
end

const ADMM_COUNTED_CG_STEPS = 20

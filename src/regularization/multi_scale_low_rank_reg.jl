"""
	MultiScaleLowRank(λ; block_sizes, time_dim=nothing, coil_dim=:coil, weights=nothing, shift=:none, rng=Random.default_rng())

Create a multi-scale locally low-rank regularization term: the same block-wise nuclear norm as
[`LocallyLowRank`](@ref), but evaluated over several block sizes at once so that both fine, localized
dynamics and large, globally correlated dynamics are penalized by the same term.

# Arguments
- `λ`: Regularization parameter: a scalar (the same threshold at every scale) or a vector with one
  entry per scale (`block_sizes`). `λ` and `weights` are not interchangeable: `λⱼ` is the nuclear-norm
  threshold *of* scale `j`'s block-decomposed term, while `weights` is the convex combination the
  proximal average takes *across* scales (see Notes) -- scaling one does not have the same effect as
  scaling the other. Use a per-scale `λ` to give coarse/fine scales different thresholds directly;
  use `weights` to bias the average toward one scale while keeping every scale's own threshold equal.
- `block_sizes`: Collection of block edge lengths, one entry per scale. Each entry is either an `Integer`
  (same edge in every spatial dimension) or a tuple with one entry per spatial dimension, as for
  [`LocallyLowRank`](@ref). A geometric progression such as `(4, 8, 16)` is the usual choice.
- `time_dim`: (optional) Dimension holding the frames / contrasts, as for [`LocallyLowRank`](@ref).
- `coil_dim`: (optional) Coil axis of an image without sensitivity maps, as for [`LocallyLowRank`](@ref):
  at every scale, each block holds all coils.
- `weights`: (optional) Convex weights of the scales; defaults to uniform. Must be non-negative and sum to 1.
- `shift`, `rng`: (optional) Grid-shift policy, as for [`LocallyLowRank`](@ref). Applied to every scale.

# Notes
- Multi-scale low-rank modelling of dynamic images was introduced by Ong & Lustig, *Beyond low rank + sparse:
  Multiscale low rank matrix decomposition*, IEEE J Sel Top Signal Process 2016, and is available in BART as
  `-R M`. Their formulation decomposes the image into one *separate component per scale*
  (`x = ∑_j x_j`, each `x_j` penalized at its own block size), which is an exact model with an exact prox per
  component; that formulation is expressible directly with this package's image decomposition — one
  [`Component`](@ref) per scale, each carrying a [`LocallyLowRank`](@ref) — and is the better choice when the
  separated components are themselves of interest.
- This term instead penalizes a *single* image at several scales at once, which has no closed-form proximal
  operator, so it uses the proximal average of the per-scale block nuclear norms (see
  [`ProximalAverage`](@ref)). This is an approximation of the sum of the per-scale penalties, not the sum
  itself: the reported objective value is the weighted average of the per-scale penalties and the minimizer
  is that of the proximal average. Prefer it when one regularized image (not a decomposition) is wanted.
- With a single entry in `block_sizes` and weight 1, the term reduces exactly to [`LocallyLowRank`](@ref).
- Cost is the sum of the per-scale costs: one SVD per block per scale and iteration.
"""
struct MultiScaleLowRank{T, B, D, C, W, RNG} <: Regularization
    λ::T
    block_sizes::B
    time_dim::D
    coil_dim::C
    weights::W
    shift::Symbol
    rng::RNG
    function MultiScaleLowRank(
            λ::T; block_sizes::B, time_dim::D = nothing, coil_dim::C = :coil, weights::W = nothing,
            shift::Symbol = :none, rng::RNG = Random.default_rng()
        ) where {T, B, D, C, W, RNG}
        @argcheck λ isa Real || λ isa AbstractVector{<:Real} "MultiScaleLowRank requires λ to be a scalar or a vector of reals, one per scale"
        @argcheck !isempty(block_sizes) "block_sizes must contain at least one scale"
        if λ isa AbstractVector
            @argcheck length(λ) == length(block_sizes) "one λ is required per scale when λ is a vector, got $(length(λ)) for $(length(block_sizes)) scales"
        end
        for block_size in block_sizes
            @argcheck block_size isa Integer || block_size isa Tuple{Vararg{Integer}} "every entry of block_sizes must be an Integer or a tuple of Integers"
            @argcheck all(block_size .> 0) "block sizes must be positive"
        end
        if weights !== nothing
            @argcheck length(weights) == length(block_sizes) "one weight is required per scale"
        end
        @argcheck shift in (:none, :fixed, :random) "shift must be :none, :fixed or :random, got :$shift"
        _check_dim_spec(time_dim, "time_dim")
        _check_dim_spec(coil_dim, "coil_dim")
        return new{T, B, D, C, W, RNG}(λ, block_sizes, time_dim, coil_dim, weights, shift, rng)
    end
end

get_operator(::MultiScaleLowRank, x::AbstractArray; threaded::Bool = true) = identity_operator(x)

function get_affected_dims(reg::MultiScaleLowRank, ::Nothing, image_dims)
    # As for LocallyLowRank, the blocks couple every dimension up to and including the temporal one.
    return image_dims[1:get_time_dim(reg.time_dim, image_dims)]
end

# Each scale is a nuclear norm, homogeneous of degree 1, and so is their average: λ scales linearly
# (elementwise when λ is a per-scale vector).
function scale_regularization(reg::MultiScaleLowRank, factor::Real)
    return MultiScaleLowRank(
        reg.λ .* factor;
        block_sizes = reg.block_sizes, time_dim = reg.time_dim, coil_dim = reg.coil_dim,
        weights = reg.weights, shift = reg.shift, rng = reg.rng
    )
end

function bind_dimensions(reg::MultiScaleLowRank, image_dims)
    return MultiScaleLowRank(
        reg.λ;
        block_sizes = reg.block_sizes,
        time_dim = get_time_dim(reg.time_dim, image_dims),
        coil_dim = _resolve_coil_dim(reg.coil_dim, image_dims),
        weights = reg.weights,
        shift = reg.shift,
        rng = reg.rng,
    )
end

function _mslr_weights(reg::MultiScaleLowRank, ::Type{R}) where {R <: Real}
    n = length(reg.block_sizes)
    reg.weights === nothing && return fill(R(1 / n), n)
    return R.(collect(reg.weights))
end

# One λ per scale: a scalar `reg.λ` is the same threshold at every scale, a vector is used as given.
function _mslr_lambdas(reg::MultiScaleLowRank, ::Type{R}) where {R <: Real}
    reg.λ isa AbstractVector && return R.(collect(reg.λ))
    return fill(R(reg.λ), length(reg.block_sizes))
end

function materialize(reg::MultiScaleLowRank, x::Variable{T}; threaded::Bool) where {T}
    x_val = ~x
    dims = dims_of(x_val)
    time_dim = get_time_dim(reg.time_dim, dims)
    @argcheck time_dim > 1 "MultiScaleLowRank needs at least one spatial dimension before the temporal one"
    spatial_size = NTuple{time_dim - 1, Int}(size(x_val)[1:(time_dim - 1)])
    num_batch = prod(size(x_val)[(time_dim + 1):end]; init = 1)
    λs = _mslr_lambdas(reg, real(T))
    coil_dim = _resolve_coil_dim(reg.coil_dim, dims)
    scales = Tuple(
        BlockNuclearNorm(
            λ_j, _llr_block_size(block_size, spatial_size, coil_dim),
            spatial_size, size(x_val, time_dim), num_batch, threaded;
            shift = reg.shift, rng = reg.rng
        ) for (λ_j, block_size) in zip(λs, reg.block_sizes)
    )
    f = ProximalAverage(scales, _mslr_weights(reg, real(T)))
    op = get_operator(reg, x_val; threaded)
    scale_repr = join((@sprintf("%g@%s", λ_j, string(s.block_size)) for (λ_j, s) in zip(λs, scales)), ", ")
    repr = @sprintf "avg_{(λ,b) ∈ {%s}} ∑ λ ⋅ ‖𝓧_b(%s)‖_*" scale_repr get_name(x)
    return StructuredOptimization.Term(1, f, op * x, repr)
end

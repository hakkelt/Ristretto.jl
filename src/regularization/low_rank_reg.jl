"""
    LowRank(λ; time_dim=nothing)

Create a low-rank regularization term with parameter `λ`. The regularization term is given by `λ‖𝓧‖_*`,
where `𝓧` is the Casorati matrix formed by unfolding the input variable
`x` along the temporal dimension specified by `time_dim`. If `time_dim`
is not provided, it will be inferred as the dimension named `:time` if `x` is a `NamedDimsArray`.

# Arguments
- `λ`: Regularization parameter, can be a scalar or an array of the same size as `x`.
- `time_dim`: (optional) Dimension along which to form the Casorati matrix. Can be an `Integer` (1-based index)
or a `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is a `NamedDimsArray`.

# Notes:
- The nuclear norm `‖𝓧‖_*` is the sum of the singular values of the matrix `𝓧`.
- The Casorati matrix is formed by reshaping `x` such that the specified `time_dim` becomes the second dimension,
and all other dimensions before it are reshaped into columns (first dimension), while dimensions after it are treated as batch dimensions.
"""
struct LowRank{T, D} <: Regularization
    λ::T
    time_dim::D
    function LowRank(λ::T; time_dim::D = nothing) where {T, D}
        _check_dim_spec(time_dim, "time_dim")
        return new{T, D}(λ, time_dim)
    end
end

"""
    RankLimit(max_rank; time_dim=nothing)

Create a rank limit regularization term that constrains the rank of the Casorati matrix formed by unfolding
the input variable `x` along the temporal dimension specified by `time_dim`. The rank of the Casorati matrix
will be limited to `max_rank`. If `time_dim` is not provided, it will be inferred as the dimension named `:time`
if `x` is a `NamedDimsArray`.

# Arguments
- `max_rank`: Maximum allowed rank for the Casorati matrix.
- `time_dim`: (optional) Dimension along which to form the Casorati matrix. Can be an `Integer` (1-based index)
or a `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is
a `NamedDimsArray`.

# Notes:
- The nuclear norm `‖𝓧‖_*` is the sum of the singular values of the matrix `𝓧`.
- The Casorati matrix is formed by reshaping `x` such that the specified `time_dim` becomes the second dimension,
and all other dimensions before it are reshaped into columns (first dimension), while dimensions after it are treated as batch dimensions.
"""
struct RankLimit{D} <: Regularization
    max_rank::Int
    time_dim::D
    function RankLimit(max_rank::Int; time_dim::D = nothing) where {D}
        @argcheck max_rank > 0 "max_rank must be positive"
        _check_dim_spec(time_dim, "time_dim")
        return new{D}(max_rank, time_dim)
    end
end

function get_operator(reg::Union{LowRank, RankLimit}, x::AbstractArray; threaded::Bool = true)
    dims = dims_of(x)
    affected_dims = 1:get_time_dim(reg.time_dim, dims)
    batch_dims = (length(affected_dims) + 1):ndims(x)
    @argcheck length(batch_dims) >= 0 "Input variable must have at least as many dimensions as affected_dims"
    if isempty(batch_dims)
        ℰ = Eye(unname(x))
    else
        x_view = @view x[fill(:, length(affected_dims))..., (ones(Int, length(batch_dims))...)]
        ℰ = Eye(unname(x_view))
    end
    ℛ = Reshape(ℰ, prod(size(x)[affected_dims[1:(end - 1)]]), size(x)[affected_dims[end]]) # Collapse all but time_dim
    if !isempty(batch_dims)
        ℛ = BatchOp(ℛ, size(x)[(length(affected_dims) + 1):end]; threaded)
    end
    if x isa NamedDimsArray
        transformed_dimnames = (:space, dimnames(x, affected_dims[end]))
        if !isempty(batch_dims)
            transformed_dimnames = (transformed_dimnames..., dimnames(x)[(length(affected_dims) + 1):end]...)
        end
        ℛ = NamedDimsOp{dimnames(x), transformed_dimnames}(ℛ)
    end
    return ℛ
end

function materialize(reg::LowRank, x::Variable{T}; threaded::Bool) where {T}
    R = real(T)
    λ = R.(reg.λ)
    op = get_operator(reg, ~x; threaded)
    repr = if reg.λ isa AbstractArray
        "λ ⋅ ‖Γ .* (ℛ * $(get_name(x)))‖_*"
    else
        @sprintf "%g ⋅ ‖ℛ * %s‖_*" λ get_name(x)
    end
    return StructuredOptimization.Term(1, _per_casorati(NuclearNorm(λ), op), op * x, repr)
end

function materialize(reg::RankLimit, x::Variable{T}; threaded::Bool) where {T}
    op = get_operator(reg, ~x; threaded)
    repr = @sprintf "rank(ℛ * %s) ≤ %d" get_name(x) reg.max_rank
    return StructuredOptimization.Term(1, _per_casorati(IndBallRank(reg.max_rank), op), op * x, repr)
end

"""
    _per_casorati(f, op)

`f` (a function of one Casorati matrix) applied to every Casorati matrix in the codomain of `op`.
Without batch dimensions the codomain is the single `(space, time)` matrix and `f` is returned as
is. With batch dimensions it is `(space, time, batch...)`, and the penalty is the sum of `f` over
the `(space, time)` slices -- one matrix per batch index, as the docstring promises -- rather than
`f` of the whole array, which neither the nuclear norm nor the rank ball defines.
"""
function _per_casorati(f, op)
    return length(size(op, 1)) == 2 ? f : CasoratiStack(f)
end

"""
    CasoratiStack(f)

`∑ₖ f(X[:, :, k...])`: the matrix function `f` applied to every `(space, time)` slice of a
`(space, time, batch...)` array. Its prox thresholds the slices one after another, each copied
into a dense matrix first. That copy is the point of the type: a `SlicedSeparableSum` would hand
`f` views, and `NuclearNorm`/`IndBallRank` route a view through their host fallback, which then
trips over the buffers they preallocated for the view type. The factorization of each slice is
threaded by `f` itself.
"""
struct CasoratiStack{F, B}
    f::F
    buf::B          # `nothing`, or `(; X, Y)`: one dense slice in, one out
end
CasoratiStack(f) = CasoratiStack(f, nothing)

for trait in (:is_convex, :is_proximable, :is_set_indicator, :is_positively_homogeneous)
    @eval ProximalCore.$trait(::Type{<:CasoratiStack{F}}) where {F} = ProximalCore.$trait(F)
end

_slice_prototype(X) = similar(X, size(X, 1), size(X, 2))
function ProximalOperators.preallocate(g::CasoratiStack, X::AbstractArray)
    S = _slice_prototype(X)
    return CasoratiStack(ProximalOperators.preallocate(g.f, S), (; X = S, Y = similar(S)))
end
_slice_buffers(g::CasoratiStack{<:Any, Nothing}, X) = (S = _slice_prototype(X); (; X = S, Y = similar(S)))
_slice_buffers(g::CasoratiStack, X) = g.buf

function (g::CasoratiStack)(X::AbstractArray)
    b = _slice_buffers(g, X)
    return sum(CartesianIndices(size(X)[3:end])) do I
        copyto!(b.X, view(X, :, :, Tuple(I)...))
        g.f(b.X)
    end
end

function ProximalCore.prox!(Y, g::CasoratiStack, X, gamma)
    b = _slice_buffers(g, X)
    value = zero(real(eltype(X)))
    for I in CartesianIndices(size(X)[3:end])
        copyto!(b.X, view(X, :, :, Tuple(I)...))
        value += prox!(b.Y, g.f, b.X, gamma)
        copyto!(view(Y, :, :, Tuple(I)...), b.Y)
    end
    return value
end

function get_affected_dims(reg::Union{LowRank, RankLimit}, ::Nothing, image_dims)
    # get_time_dim returns an index; return the corresponding entries of image_dims so
    # that named dimensions stay Symbols (needed for task-splitting setdiff).
    time_dim = get_time_dim(reg.time_dim, image_dims)
    return image_dims[1:time_dim]
end

# Nuclear norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
scale_regularization(reg::LowRank, factor::Real) = LowRank(reg.λ .* factor; time_dim = reg.time_dim)
# Rank constraints are scale-invariant (rank(factor * X) == rank(X)); no correction needed.
scale_regularization(reg::RankLimit, factor::Real) = reg

bind_dimensions(reg::LowRank, image_dims) = LowRank(reg.λ; time_dim = get_time_dim(reg.time_dim, image_dims))
bind_dimensions(reg::RankLimit, image_dims) = RankLimit(reg.max_rank; time_dim = get_time_dim(reg.time_dim, image_dims))

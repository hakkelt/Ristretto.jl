export MatrixOp

"""
	MatrixOp(domain_type=Float64::Type, dim_in::Tuple, A::AbstractMatrix; threaded=true)
	MatrixOp(A::AbstractMatrix; threaded=true)
	MatrixOp(A::AbstractMatrix, n_colons; threaded=true)

Creates a `LinearOperator` which, when multiplied with a vector `x::AbstractVector`, returns the product `A*x`.

The input `x` can be also a matrix: the number of columns must be given either in the second entry of `dim_in::Tuple` or using the constructor `MatrixOp(A::AbstractMatrix, n_colons)`.

- `threaded`: `false` forces `mul!` to run with BLAS restricted to a single thread; `true`
  (default) leaves BLAS at whatever thread budget is in force, including any outer
  `NestedThreading` restriction. BLAS reads its thread count live on every call (there is
  no plan to bake it into), so this is a per-call scope rather than a construction-time
  choice.

  There is deliberately no size threshold on the `true` branch: BLAS owns its own thread
  pool and already applies its own per-call size heuristic, so a threshold here would only
  second-guess it from less information (`length(A)` rather than the real `m*n*k`). It
  follows that `is_threaded` reports the *permission* -- "BLAS may use its own budget" --
  and not a promise that any particular `gemv`/`gemm` was actually split.

```jldoctest
julia> MatrixOp(Float64,(10,),randn(20,10))
▒  ℝ^10 -> ℝ^20

julia> MatrixOp(randn(20,10))
▒  ℝ^10 -> ℝ^20

julia> MatrixOp(Float64,(10,20),randn(20,10))
▒  ℝ^(10, 20) -> ℝ^(20, 20)

julia> MatrixOp(randn(20,10),4)
▒  ℝ^(10, 4) -> ℝ^(20, 4)

```
"""
struct MatrixOp{D, T, M <: AbstractMatrix{T}, NC, dS, cS} <: LinearOperator
    A::M
    threaded::Bool
end

# Constructors

###standard constructor Operator{N}(domain_type::Type, DomainDim::NTuple{N,Int})
function _matrixop_impl(
        domain_type::Type, DomainDim::NTuple{N, Int}, A::M,
        array_type::Type{<:AbstractArray}, threaded::Bool,
    ) where {N, T, M <: AbstractMatrix{T}}
    N > 2 && error("cannot multiply a Matrix by a n-dimensional Variable with n > 2")
    size(A, 2) != DomainDim[1] && error("wrong input dimensions")
    codomainT = domain_type <: Real && T <: Complex ? T : domain_type
    dS = _normalize_array_type(array_type, domain_type)
    cS = _normalize_array_type(array_type, codomainT)
    th = _blas_threaded(threaded, dS)
    return if N == 1
        MatrixOp{domain_type, T, M, 1, dS, cS}(A, th)
    else
        MatrixOp{domain_type, T, M, DomainDim[2], dS, cS}(A, th)
    end
end

function MatrixOp(
        domain_type::Type, DomainDim::NTuple{N, Int}, A::M
        ; array_type::Type{<:AbstractArray} = _array_wrapper_type(M), threaded::Bool = true,
    ) where {N, M <: AbstractMatrix}
    return _matrixop_impl(domain_type, DomainDim, A, array_type, threaded)
end
###

function MatrixOp(
        A::M; array_type::Type{<:AbstractArray} = _array_wrapper_type(M), threaded::Bool = true
    ) where {M <: AbstractMatrix}
    return _matrixop_impl(eltype(A), (size(A, 2),), A, array_type, threaded)
end
function MatrixOp(
        D::Type, A::M; array_type::Type{<:AbstractArray} = _array_wrapper_type(M), threaded::Bool = true
    ) where {M <: AbstractMatrix}
    return _matrixop_impl(D, (size(A, 2),), A, array_type, threaded)
end
function MatrixOp(A::M, n::Integer; threaded::Bool = true) where {M <: AbstractMatrix}
    return _matrixop_impl(eltype(A), (size(A, 2), n), A, _array_wrapper_type(M), threaded)
end
function MatrixOp(D::Type, A::M, n::Integer; threaded::Bool = true) where {M <: AbstractMatrix}
    return _matrixop_impl(D, (size(A, 2), n), A, _array_wrapper_type(M), threaded)
end

function Scale(coeff::Number, A::MatrixOp{D, T, M, NC, dS, cS}) where {D, T, M, NC, dS, cS}
    if coeff == 1
        return A
    end
    cT = codomain_type(A)
    isCodomainReal = typeof(cT) <: Tuple ? all([t <: Real for t in cT]) : cT <: Real
    if isCodomainReal && typeof(coeff) <: Complex
        error(
            "Cannot Scale AbstractOperator with real codomain with complex scalar. Use `DiagOp` instead.",
        )
    end
    return MatrixOp(domain_type(A), size(A, 2), coeff * A.A; array_type = _array_wrapper_type(dS), threaded = A.threaded)
end
function Scale(coeff::Number, A::AdjointOperator{<:MatrixOp})
    if coeff == 1
        return A
    end
    cT = codomain_type(A)
    isCodomainReal = typeof(cT) <: Tuple ? all([t <: Real for t in cT]) : cT <: Real
    if isCodomainReal && typeof(coeff) <: Complex
        error(
            "Cannot Scale AbstractOperator with real codomain with complex scalar. Use `DiagOp` instead.",
        )
    end
    return AdjointOperator(Scale(conj(coeff), A.A))
end

import Base: convert
convert(::Type{LinearOperator}, L::M) where {T, M <: AbstractMatrix{T}} = MatrixOp(T, (size(L, 2),), L)
function convert(::Type{LinearOperator}, L::M, n::Integer) where {T, M <: AbstractMatrix{T}}
    return MatrixOp(L, n)
end
function convert(::Type{LinearOperator}, dom::Type, dim_in::Tuple, L::AbstractMatrix)
    return MatrixOp(dom, dim_in, L)
end

# Mappings

function mul!(y::AbstractArray, L::MatrixOp, b::AbstractArray)
    check(y, L, b)
    return _with_blas_threading(L.threaded, domain_array_type(L), _matmul_work(L.A, b)) do
        mul!(y, L.A, b)
    end
end
# NC=1 implicit batching: matrix input accepted (domain declared as 1D but each column is processed)
function mul!(y::AbstractArray, L::MatrixOp{<:Any, <:Any, <:Any, 1}, b::AbstractArray)
    return _with_blas_threading(L.threaded, domain_array_type(L), _matmul_work(L.A, b)) do
        mul!(y, L.A, b)
    end
end
function mul!(y::AbstractArray, L::AdjointOperator{<:MatrixOp}, b::AbstractArray)
    check(y, L, b)
    return _with_blas_threading(L.A.threaded, domain_array_type(L.A), _matmul_work(L.A.A', b)) do
        mul!(y, L.A.A', b)
    end
end
# NC=1 adjoint implicit batching: matrix input accepted
function mul!(
        y::AbstractArray, L::AdjointOperator{<:MatrixOp{<:Any, <:Any, <:Any, 1}}, b::AbstractArray
    )
    return _with_blas_threading(L.A.threaded, domain_array_type(L.A), _matmul_work(L.A.A', b)) do
        mul!(y, L.A.A', b)
    end
end

# Special Case, real b, complex matrix: accepts real b (type mismatch is intentional)
function mul!(y::AbstractArray, L::AdjointOperator{<:MatrixOp{D, T}}, b::AbstractArray) where {D <: Real, T <: Complex}
    check(y, L, b)
    yc = similar(y, T, size(y))
    _with_blas_threading(L.A.threaded, domain_array_type(L.A), _matmul_work(L.A.A', b)) do
        mul!(yc, L.A.A', b)
    end
    return y .= real.(yc)
end
# Resolves ambiguity: NC=1 real-domain complex-matrix adjoint with matrix batching
function mul!(y::AbstractArray, L::AdjointOperator{<:MatrixOp{D, T, M, 1, <:Any, <:Any}}, b::AbstractArray) where {D <: Real, T <: Complex, M}
    yc = similar(y, T, size(y))
    _with_blas_threading(L.A.threaded, domain_array_type(L.A), _matmul_work(L.A.A', b)) do
        mul!(yc, L.A.A', b)
    end
    return y .= real.(yc)
end

function mul!(y::AbstractArray, L::MatrixOp, b::AbstractArray, α::Number, β::Number)
    check(y, L, b)
    return _with_blas_threading(L.threaded, domain_array_type(L)) do
        mul!(y, L.A, b, _coefficient(y, α), _coefficient(y, β))
    end
end
function mul!(y::AbstractArray, L::MatrixOp{<:Any, <:Any, <:Any, 1}, b::AbstractArray, α::Number, β::Number)
    return _with_blas_threading(L.threaded, domain_array_type(L)) do
        mul!(y, L.A, b, _coefficient(y, α), _coefficient(y, β))
    end
end
function mul!(y::AbstractArray, L::AdjointOperator{<:MatrixOp}, b::AbstractArray, α::Number, β::Number)
    check(y, L, b)
    return _with_blas_threading(L.A.threaded, domain_array_type(L.A)) do
        mul!(y, L.A.A', b, _coefficient(y, α), _coefficient(y, β))
    end
end
function mul!(
        y::AbstractArray, L::AdjointOperator{<:MatrixOp{<:Any, <:Any, <:Any, 1}}, b::AbstractArray, α::Number, β::Number
    )
    return _with_blas_threading(L.A.threaded, domain_array_type(L.A)) do
        mul!(y, L.A.A', b, _coefficient(y, α), _coefficient(y, β))
    end
end
function mul!(
        y::AbstractArray, L::AdjointOperator{<:MatrixOp{D, T}}, b::AbstractArray, α::Number, β::Number
    ) where {D <: Real, T <: Complex}
    check(y, L, b)
    yc = similar(y, T, size(y))
    _with_blas_threading(L.A.threaded, domain_array_type(L.A)) do
        mul!(yc, L.A.A', b)
    end
    return _store!(y, Broadcast.broadcasted(real, yc), α, β)
end
function mul!(
        y::AbstractArray, L::AdjointOperator{<:MatrixOp{D, T, M, 1, <:Any, <:Any}}, b::AbstractArray, α::Number, β::Number
    ) where {D <: Real, T <: Complex, M}
    yc = similar(y, T, size(y))
    _with_blas_threading(L.A.threaded, domain_array_type(L.A)) do
        mul!(yc, L.A.A', b)
    end
    return _store!(y, Broadcast.broadcasted(real, yc), α, β)
end

# Properties

domain_type(::MatrixOp{D}) where {D} = D
codomain_type(::MatrixOp{D, T}) where {D, T} = D <: Real && T <: Complex ? T : D
domain_array_type(::MatrixOp{D, T, M, NC, dS}) where {D, T, M, NC, dS} = dS
codomain_array_type(::MatrixOp{D, T, M, NC, dS, cS}) where {D, T, M, NC, dS, cS} = cS
is_thread_safe(::MatrixOp) = true
is_threaded(L::MatrixOp) = L.threaded
supports_threading(::MatrixOp) = true

# Type-stable size dispatch: NC=1 → 1D, NC>1 → 2D
size(L::MatrixOp{D, T, M, 1, dS, cS}) where {D, T, M, dS, cS} = ((size(L.A, 1),), (size(L.A, 2),))
size(L::MatrixOp{D, T, M, NC, dS, cS}) where {D, T, M, NC, dS, cS} = ((size(L.A, 1), NC), (size(L.A, 2), NC))

fun_name(L::MatrixOp) = "▒"

is_diagonal(L::MatrixOp) = isdiag(L.A)
is_symmetric(L::MatrixOp) = issymmetric(L.A)
is_AAc_diagonal(L::MatrixOp) = _gram_is_diagonal(L.A, 1)
is_AcA_diagonal(L::MatrixOp) = _gram_is_diagonal(L.A, 2)

"""
    _gram_is_diagonal(A, d) -> Bool

Whether `A * Aᴴ` (`d = 1`) or `Aᴴ * A` (`d = 2`) is diagonal, compared against exact zero as
`isdiag` does: whether the rows (`d = 1`) or columns (`d = 2`) of `A` are pairwise orthogonal.

Forming the product costs `O(m²n)` and an `m × m` temporary, more than several iterations of a
solver that asks the question while choosing a formulation. A single non-orthogonal pair disproves
the answer, so pairs among the first few rows (columns) are tested first, `O(n)` each, and a matrix
that is not genuinely Gram-diagonal, the overwhelmingly common case, is rejected there. Only one
that survives the sample reaches the full product, so the answer is always the exact one.
"""
function _gram_is_diagonal(A::AbstractMatrix, d::Int)
    _gram_sample_is_diagonal(A, d) || return false
    return _isdiag(d == 1 ? A * A' : A' * A)
end

# The pairs among the first `_GRAM_SAMPLE` rows (`d = 1`) or columns (`d = 2`) of `A`, each a
# `dot` of two views, which allocates nothing for a host array.
const _GRAM_SAMPLE = 5
function _gram_sample_is_diagonal(A::AbstractMatrix, d::Int)
    k = min(size(A, d), _GRAM_SAMPLE)
    if d == 1
        for i in 1:(k - 1), j in (i + 1):k
            iszero(dot(view(A, i, :), view(A, j, :))) || return false
        end
    else
        for i in 1:(k - 1), j in (i + 1):k
            iszero(dot(view(A, :, i), view(A, :, j))) || return false
        end
    end
    return true
end

_isdiag(G) = isdiag(G)
is_null(L::MatrixOp) = L.A == 0 * I
is_eye(L::MatrixOp) = L.A == I
function is_invertible(L::MatrixOp)
    return size(L.A, 1) == size(L.A, 2) &&
        !isapprox(det(BigFloat.(Array(L.A))), 0; atol = eps(eltype(L.A)) * 10)
end
function is_orthogonal(L::MatrixOp)
    return size(L.A, 1) == size(L.A, 2) && all(<(eps(eltype(L.A)) * 10), L.A' * L.A - I)
end
# `rank`/`det`/`eigvals` are SVD/LAPACK-bound: no generic GPU implementation exists, so
# these always run on a host copy (cheap relative to the O(n^3) decomposition itself, and
# these predicates are constructor-time metadata, not something called per solver iteration).
is_full_row_rank(L::MatrixOp) = rank(Array(L.A)) == size(L.A, 1)
is_full_column_rank(L::MatrixOp) = rank(Array(L.A)) == size(L.A, 2)
is_positive_definite(L::MatrixOp) = isposdef(Array(L.A))
is_positive_semidefinite(L::MatrixOp) = issymmetric(L.A) && all(eigvals(Symmetric(Array(L.A))) .>= 0)

has_optimized_normalop(::MatrixOp) = true
get_normal_op(L::MatrixOp) = MatrixOp(domain_type(L), size(L, 2), L.A' * L.A; threaded = L.threaded)

has_fast_opnorm(::MatrixOp) = true
LinearAlgebra.opnorm(L::MatrixOp) = opnorm(L.A)

function _copy_operator_impl(
        op::MatrixOp{D, T, M, NC, dS, cS}; storage_type = nothing, threaded = nothing
    ) where {D, T, M, NC, dS, cS}
    new_threaded = threaded === nothing ? op.threaded : threaded
    # `op.A` is read-only operator data, so it is shared rather than copied unless the
    # storage backend actually has to change.
    new_at = storage_type === nothing ? _array_wrapper_type(dS) : storage_type
    new_A = storage_type === nothing ? op.A : similar(storage_type{T}, size(op.A)) .= op.A
    return MatrixOp(D, size(op, 2), new_A; array_type = new_at, threaded = new_threaded)
end

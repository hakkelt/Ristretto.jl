export AdjointOperator

"""
	AdjointOperator(A::AbstractOperator)

Shorthand constructor:

	'(A::AbstractOperator)

Returns the adjoint operator of `A`.

```jldoctest
julia> AdjointOperator(ZeroPad((2,2),(0,2)))
[I;0]ᵃ  ℝ^(2, 4) -> ℝ^(2, 2)

julia> [Eye(10); FiniteDiff((10,))]'
[I;δx]ᵃ  ℝ^10  ℝ^9 -> ℝ^10
	
```
"""
struct AdjointOperator{T <: AbstractOperator} <: AbstractOperator
    A::T
    function AdjointOperator(A::T) where {T <: AbstractOperator}
        # The adjoint of an affine operator is the adjoint of its linear part.
        is_affine(A) == false &&
            error("Cannot transpose a nonlinear operator. You might use `jacobian`")
        return new{T}(A)
    end
end

_ndoms_from_type(::Type{<:AdjointOperator{T}}, dim::Int) where {T} = _ndoms_from_type(T, dim == 1 ? 2 : 1)

# Constructors

AdjointOperator(L::AdjointOperator) = L.A

# Properties

has_fast_opnorm(L::AdjointOperator) = has_fast_opnorm(L.A)
LinearAlgebra.opnorm(L::AdjointOperator) = opnorm(L.A)
opnorm_bound(L::AdjointOperator) = opnorm_bound(L.A)
# The keywords are forwarded, not swallowed: `‖A'‖ = ‖A‖` exactly, so whatever promise the inner
# call keeps about its own margin and side is the promise this call makes.
estimate_opnorm(L::AdjointOperator; kwargs...) = estimate_opnorm(L.A; kwargs...)

Base.:(==)(L1::AdjointOperator{T}, L2::AdjointOperator{T}) where {T} = L1.A == L2.A
size(L::AdjointOperator) = size(L.A, 2), size(L.A, 1)

domain_type(L::AdjointOperator) = codomain_type(L.A)
codomain_type(L::AdjointOperator) = domain_type(L.A)
domain_array_type(L::AdjointOperator) = codomain_array_type(L.A)
codomain_array_type(L::AdjointOperator) = domain_array_type(L.A)
is_thread_safe(L::AdjointOperator) = is_thread_safe(L.A)

fun_name(L::AdjointOperator) = fun_name(L.A) * "ᵃ"

is_linear(L::AdjointOperator) = is_affine(L.A)
is_null(L::AdjointOperator) = is_null(L.A)
is_eye(L::AdjointOperator) = is_eye(L.A)
is_diagonal(L::AdjointOperator) = is_diagonal(L.A)
is_AcA_diagonal(L::AdjointOperator) = is_AAc_diagonal(L.A)
is_AAc_diagonal(L::AdjointOperator) = is_AcA_diagonal(L.A)
is_orthogonal(L::AdjointOperator) = is_orthogonal(L.A)
is_invertible(L::AdjointOperator) = is_invertible(L.A)
is_full_row_rank(L::AdjointOperator) = is_full_column_rank(L.A)
is_full_column_rank(L::AdjointOperator) = is_full_row_rank(L.A)

diag(L::AdjointOperator) = diag(L.A)
diag_AcA(L::AdjointOperator) = diag_AAc(L.A)
diag_AAc(L::AdjointOperator) = diag_AcA(L.A)

_children(L::AdjointOperator) = (L.A,)

function _copy_operator_impl(L::AdjointOperator; storage_type = nothing, threaded = nothing)
    return AdjointOperator(copy_operator(L.A; storage_type, threaded))
end

"""
	add_mul!(y, L::AbstractOperator, b, buf, α = true, β = true)

`y = α * (L * b) + β * y`, with `buf`, an array of the codomain of `L`, as the place where
`L * b` may be written: the generic method writes it there and combines it with `y` in a second
pass. An operator whose 5-argument `mul!` accumulates into `y` directly uses that and ignores
`buf`.

Internal, not exported. The combinators sum their blocks through it: `VCAT`'s adjoint does, so a
stack of `N` blocks that each touch a small, disjoint part of the domain — a per-frame `GetIndex`
stack, say — is quadratic in `N` without an accumulating block and linear with one. Measured on a
`128×128×T` domain with one `GetIndex` per frame, adjoint wall time went 0.81 → 0.13 ms at
`T = 4` and 162.6 → 2.9 ms at `T = 64`.
"""
function add_mul!(y::AbstractArray, L::AbstractOperator, b, buf::AbstractArray, α::Number = true, β::Number = true)
    mul!(buf, L, b)
    return _store!(y, buf, α, β)
end

# `dest = α * src + β * dest`, for `src` an array or a lazy `Broadcasted`, in one pass and on
# `Th` threads; `dest` is not read when `β` is zero. The coefficients are converted to the
# precision of `dest`, so that a `Float64` coefficient does not widen a `Float32` loop. The
# combination goes through FastBroadcast, which vectorizes a lazy `src` over views where Base's
# broadcast does not, and runs over the flat arrays when all of them are `Array`s; for other array
# styles it falls back to Base's broadcast. Constant propagation is off: the arguments' types
# already fix the kernel, and the semi-concrete evaluation of the views a caller builds from
# constant ranges makes JET 0.11 fail on Julia 1.13 with a `TypeError` in its own inference.
@inline Base.@constprop :none function _store!(dest, src, α::Number, β::Number, ::Val{Th} = Val(false)) where {Th}
    a, b = _coefficient(dest, α), _coefficient(dest, β)
    return _store_fast!(dest, src, a, b, _fbthread(_fbbool(Th)))
end

_coefficient(dest, c::Bool) = c
_coefficient(dest, c::Real) = convert(real(eltype(dest)), c)
_coefficient(dest, c::Number) = convert(eltype(dest), c)

@inline function _store_fast!(dest, src, α, β, th)
    if iszero(β)
        if isone(α)
            @.. thread = th dest = identity(src)
        else
            @.. thread = th dest = α * src
        end
    elseif isone(α)
        @.. thread = th dest = src + β * dest
    else
        @.. thread = th dest = α * src + β * dest
    end
    return dest
end

# `y = β * y`; `y` is not read when `β` is zero.
function _scale_output!(y, β::Number)
    if iszero(β)
        fill!(y, zero(eltype(y)))
    elseif !isone(β)
        b = _coefficient(y, β)
        @.. y = b * y
    end
    return y
end

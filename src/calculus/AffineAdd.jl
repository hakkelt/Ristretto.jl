export AffineAdd

"""
	AffineAdd(A::AbstractOperator, d, [sign = true])

Affine addition to `AbstractOperator` with an array or scalar `d`.

Use `sign = false` to perform subtraction.

```jldoctest
julia> A = AffineAdd(Sin(3),[1.;2.;3.])
sin+d  ℝ^3 -> ℝ^3

julia> A*[3.;4.;5.] == sin.([3.;4.;5.]).+[1.;2.;3.]
true

julia> A = AffineAdd(Exp(3),[1.;2.;3.],false)
e-d  ℝ^3 -> ℝ^3

julia> A*[3.;4.;5.] == exp.([3.;4.;5.]).-[1.;2.;3.]
true
	
```
"""
struct AffineAdd{L <: AbstractOperator, D <: Union{AbstractArray, Number}, S} <: AbstractOperator
    A::L
    d::D
    function AffineAdd(
            A::L, d::D, sign::Bool = true
        ) where {L <: AbstractOperator, D <: AbstractArray}
        # `size(d)` is `d`'s flat Base.size, which only matches `size(A, 1)`
        # (a nested per-block tuple for a multi-domain codomain) when the
        # codomain is single-block. Compare structurally instead, same as
        # `check()` in utils.jl does for domain/codomain arrays generally.
        dim_d = d isa AbstractOperators.ArrayPartition ? size.(d.x) : size(d)
        if !isequal(dim_d, size(A, 1))
            throw(
                DimensionMismatch(
                    "codomain size of $A not compatible with array `d` of size $(dim_d)"
                ),
            )
        end
        # Same structural-vs-flat mismatch as the size check above:
        # `codomain_type(A)` is a nested Tuple of per-block types for a
        # multi-domain codomain, while `eltype` of an ArrayPartition is its
        # single shared scalar type.
        dtype_d = d isa AbstractOperators.ArrayPartition ? eltype.(d.x) : eltype(d)
        if !isequal(dtype_d, codomain_type(A))
            error(
                "cannot tilt opertor having codomain type $(codomain_type(A)) with array of type $(dtype_d)",
            )
        end
        return new{L, D, sign}(A, d)
    end
    # scalar
    function AffineAdd(A::L, d::D, sign::Bool = true) where {L <: AbstractOperator, D <: Number}
        if typeof(d) <: Complex && codomain_type(A) <: Real
            error(
                "cannot tilt opertor having codomain type $(codomain_type(A)) with array of type $(eltype(d))",
            )
        end
        return new{L, D, sign}(A, d)
    end
end

function Scale(coeff::Number, T::AffineAdd{L, D, S}) where {L, D, S}
    return coeff == 1 ? T : AffineAdd(Scale(coeff, T.A), coeff * T.d, S)
end

# Mappings
# array
function mul!(y::AbstractArray, T::AffineAdd{<:AbstractOperator, <:Any, true}, x::AbstractArray)
    check(y, T, x)
    mul!(y, T.A, x)
    return @.. y += T.d
end

function mul!(y::AbstractArray, T::AffineAdd{<:AbstractOperator, <:Any, false}, x::AbstractArray)
    check(y, T, x)
    mul!(y, T.A, x)
    return @.. y -= T.d
end

function mul!(y::AbstractArray, T::AdjointOperator{<:AffineAdd}, x::AbstractArray)
    check(y, T, x)
    return mul!(y, T.A.A', x)
end

# Properties

function Base.:(==)(L1::AffineAdd{L, D, S}, L2::AffineAdd{L, D, S}) where {L, D, S}
    return L1.A == L2.A && L1.d == L2.d
end
size(L::AffineAdd) = size(L.A)

domain_type(L::AffineAdd) = domain_type(L.A)
codomain_type(L::AffineAdd) = codomain_type(L.A)
domain_array_type(L::AffineAdd) = domain_array_type(L.A)
codomain_array_type(L::AffineAdd) = codomain_array_type(L.A)
is_thread_safe(L::AffineAdd) = is_thread_safe(L.A)

# Never linear: `A * 0 = d`. A zero `d` is not looked for, since that would make a type-level
# trait an O(n) scan of the displacement.
is_linear(L::AffineAdd) = false
is_affine(L::AffineAdd) = is_affine(L.A)
is_null(L::AffineAdd) = is_null(L.A) && all(displacement(L) .== 0)
is_eye(L::AffineAdd) = is_eye(L.A) && all(displacement(L) .== 0)
is_diagonal(L::AffineAdd) = is_diagonal(L.A)
is_invertible(L::AffineAdd) = is_invertible(L.A)
is_AcA_diagonal(L::AffineAdd) = is_AcA_diagonal(L.A)
is_AAc_diagonal(L::AffineAdd) = is_AAc_diagonal(L.A)
is_full_row_rank(L::AffineAdd) = is_full_row_rank(L.A)
is_full_column_rank(L::AffineAdd) = is_full_column_rank(L.A)

is_sliced(L::AffineAdd) = is_sliced(L.A)
get_slicing_expr(L::AffineAdd) = get_slicing_expr(L.A)
get_slicing_mask(L::AffineAdd) = get_slicing_mask(L.A)
remove_slicing(L::AffineAdd) = remove_slicing(L.A)

fun_name(T::AffineAdd{L, D, S}) where {L, D, S} = "$(fun_name(T.A))" * (S ? "+" : "-") * "d"

# `get_normal_op` keeps the displacement, so `powerit` iterates `x -> A'Ax + A'd` and the linear
# part alone does not bound it. Use `remove_displacement` to ask for that instead.
opnorm_bound(L::AffineAdd) = all(iszero, displacement(L)) ? opnorm_bound(L.A) : Inf

diag(L::AffineAdd) = diag(L.A)
diag_AcA(L::AffineAdd) = diag_AcA(L.A)
diag_AAc(L::AffineAdd) = diag_AAc(L.A)

has_optimized_normalop(T::AffineAdd) = has_optimized_normalop(T.A)
function get_normal_op(T::AffineAdd{L, D, S}) where {L, D, S}
    return AffineAdd(get_normal_op(T.A), T.A' * T.d, S)
end

# utils
import Base: sign
sign(T::AffineAdd{L, D, false}) where {L, D} = -1
sign(T::AffineAdd{L, D, true}) where {L, D} = 1

function permute(T::AffineAdd{L, D, S}, p::AbstractVector{Int}) where {L, D, S}
    A = permute(T.A, p)
    return AffineAdd(A, T.d, S)
end

# Around a linear operator the displacement is `d` itself, returned without a copy: like the
# displacement stored in the operator, it is read-only.
displacement(A::AffineAdd{L, D, true}) where {L, D} = is_linear(A.A) ? A.d : A.d .+ displacement(A.A)
displacement(A::AffineAdd{L, D, false}) where {L, D} = is_linear(A.A) ? -A.d : -A.d .+ displacement(A.A)

remove_displacement(A::AffineAdd) = remove_displacement(A.A)

_children(L::AffineAdd) = (L.A,)

function _copy_operator_impl(
        L::AffineAdd{Op, D, S}; storage_type = nothing, threaded = nothing
    ) where {Op, D, S}
    new_A = copy_operator(L.A; storage_type, threaded)
    # `d` is read-only displacement data, shared unless the storage backend must change.
    new_d = _copy_displacement(L.d, storage_type)
    return AffineAdd(new_A, new_d, S)
end

_copy_displacement(d::Number, ::Any) = d
_copy_displacement(d::AbstractArray, ::Nothing) = d
function _copy_displacement(d::AbstractArray, storage_type::Type{<:AbstractArray})
    return copyto!(similar(storage_type{eltype(d)}, size(d)), d)
end

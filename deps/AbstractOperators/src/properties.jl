import Base: size, ndims, similar, copy
import LinearAlgebra: diag, opnorm

export copy_operator
# _should_thread is internal, not exported
export ndoms,
    domain_type,
    codomain_type,
    domain_array_type,
    codomain_array_type,
    is_linear,
    is_affine,
    is_eye,
    is_null,
    is_diagonal,
    is_AcA_diagonal,
    is_AAc_diagonal,
    diag_AcA,
    diag_AAc,
    is_orthogonal,
    is_invertible,
    is_full_row_rank,
    is_full_column_rank,
    is_positive_definite,
    is_positive_semidefinite,
    is_symmetric,
    is_sliced,
    remove_slicing,
    displacement,
    remove_displacement,
    is_thread_safe,
    estimate_opnorm,
    opnorm_bound

"""
	domain_type(A::AbstractOperator)

Returns the type of the domain.

```jldoctest
julia> domain_type(DiagOp(rand(10)))
Float64

julia> domain_type(hcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64, 10))))
(ComplexF64, ComplexF64)
```
"""
domain_type

"""
	codomain_type(A::AbstractOperator)

Returns the type of the codomain.

```jldoctest
julia> codomain_type(DiagOp(rand(ComplexF64, 10)))
ComplexF64 (alias for Complex{Float64})

julia> codomain_type(vcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64, 10))))
(ComplexF64, ComplexF64)
```
"""
codomain_type

"""
	domain_array_type(L::AbstractOperator)

Returns the type of the storage for the domain of the operator.

```jldoctest
julia> domain_array_type(DiagOp(rand(10)))
Array{Float64}

julia> domain_array_type(hcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64, 10))))
RecursiveArrayTools.ArrayPartition{ComplexF64, Tuple{Array{ComplexF64}, Array{ComplexF64}}}
```
"""
function domain_array_type(L::AbstractOperator)
    return _storage_type_for_elem(domain_type(L))
end

"""
	codomain_array_type(L::AbstractOperator)

Returns the type of the storage of for the codomain of the operator.

```jldoctest
julia> codomain_array_type(DiagOp(rand(ComplexF64,10)))
Array{ComplexF64}

julia> codomain_array_type(vcat(Eye(Complex{Float64},(10,)),DiagOp(rand(ComplexF64,10))))
RecursiveArrayTools.ArrayPartition{ComplexF64, Tuple{Array{ComplexF64}, Array{ComplexF64}}}
```
"""
function codomain_array_type(L::AbstractOperator)
    return _storage_type_for_elem(codomain_type(L))
end

_storage_type_for_elem(T::Type) = Array{T}
function _storage_type_for_elem(dt::Tuple)
    arrayTypes = Tuple{[Array{t} for t in dt]...}
    return ArrayPartition{promote_type(dt...), arrayTypes}
end

# Get the array container type (Array, CuArray, etc.) from an array instance
_array_wrapper_type(::Type{A}) where {A <: AbstractArray} = Base.typename(A).wrapper
function _array_wrapper(x::A) where {T, A <: AbstractArray{T}}
    return _array_wrapper_type(typeof(x isa SubArray ? parent(x) : x))
end
function _normalize_array_type(array_type::Type{A}, elem_type::Type{T}) where {A <: AbstractArray, T}
    return _array_wrapper_type(A){T}
end

_storage_eltype(::Type{<:AbstractArray{T}}) where {T} = T

function allocate_in_domain(L::AbstractOperator, dims... = size(L, 2)...)
    dS = domain_array_type(L)
    if dS <: ArrayPartition
        S = dS.parameters[2]
        return ArrayPartition([similar(s, d...) for (s, d) in zip(S.parameters, dims)]...)
    elseif dS <: Array
        return Array{domain_type(L), length(dims)}(undef, dims...)
    else
        return similar(dS, dims...)
    end
end

function allocate_in_codomain(L::AbstractOperator, dims... = size(L, 1)...)
    cS = codomain_array_type(L)
    if cS <: ArrayPartition
        S = cS.parameters[2]
        return ArrayPartition([similar(s, d...) for (s, d) in zip(S.parameters, dims)]...)
    elseif cS <: Array
        return Array{codomain_type(L), length(dims)}(undef, dims...)
    else
        return similar(cS, dims...)
    end
end

array_type_display_string(::Type{T}) where {T <: AbstractArray} = ""
function storage_display_string(L::AbstractOperator)
    return array_type_display_string(codomain_array_type(L))
end

"""
	is_thread_safe(L::AbstractOperator)

Returns whether the operator is thread safe (i.e. it can be used on multiple arrays simulaneously).
"""
is_thread_safe(L::AbstractOperator) = false

"""
	size(A::AbstractOperator, [dom,])

Returns the size of an `AbstractOperator`. Type `size(A,1)` for the size of the codomain and `size(A,2)` for the size of the codomain.

Note that the size is always returned as a `Tuple`, so for a 2D operator the size of the codomain will be `(m,)`
and the size of the domain will be `(n,)` for an `m x n` operator.

```jldoctest
julia> size(FiniteDiff((10,20), 1))
((9, 20), (10, 20))

julia> size(FiniteDiff((10,20), 1),1)
(9, 20)

julia> size(FiniteDiff((10,20), 1),2)
(10, 20)
```
"""
size(L::AbstractOperator, i::Int) = size(L)[i]

# `map` over the size tuple rather than `count_dims(size(L, i))`: for operators with
# heterogeneous codomain/domain shapes the two entries have different types, so a
# non-literal index widens the result to a `Union` and makes `count_dims` a runtime dispatch.
"""
	ndims(A::AbstractOperator, [dom,])

Returns a `Tuple` with the number of dimensions of the codomain and domain of an `AbstractOperator`.  Type `ndims(A,1)` for the number of dimensions of the codomain and `ndims(A,2)` for the number of dimensions of the codomain.

```jldoctest; setup = :(using AbstractOperators)
julia> V = Variation((2,3,4))
Ʋ  ℝ^(2, 3, 4) -> ℝ^(24, 3)

julia> ndims(V)
(2, 3)

julia> ndims(V,1)
2

julia> ndims(V,2)
3
```
"""
ndims(L::AbstractOperator) = map(count_dims, size(L))
ndims(L::AbstractOperator, i::Int) = ndims(L)[i]

count_dims(::Tuple{}) = 0
count_dims(::NTuple{N, <:Integer}) where {N} = N
count_dims(dims::Tuple) = map(count_dims, dims)

"""
	ndoms(L::AbstractOperator, [dom::Int]) -> (number of codomains, number of domains)

Returns the number of codomains and domains  of a `AbstractOperator`. Optionally you can specify the codomain (with `dom = 1`) or the domain (with `dom = 2`)

```jldoctest
julia> ndoms(Eye(10,10))
(1, 1)

julia> ndoms(hcat(Eye(10,10),Eye(10,10)))
(1, 2)

julia> ndoms(hcat(Eye(10,10),Eye(10,10)),2)
2

julia> ndoms(DCAT(Eye(10,10),Eye(10,10)))
(2, 2)
```
"""
function ndoms(L::AbstractOperator)
    # Recompute from `size(L)` instead of `length.(ndims(L))`: for operators whose codomain
    # and domain shapes have different types, JET widens `ndims(L)`'s tuple across the call
    # boundary into `Tuple{Union{...}, Union{...}}` and reports the elementwise `length` as
    # runtime dispatch. Indexing the size tuple here keeps both entries concrete.
    sz = size(L)
    return length(count_dims(sz[1])), length(count_dims(sz[2]))
end
ndoms(L::AbstractOperator, i::Int) = ndoms(L)[i]

"""
	is_linear(A::AbstractOperator)

Returns true if `A` is linear: `A * (αx + βy) = α(A * x) + β(A * y)`, so `A * 0 = 0`. An
operator with a displacement (`AffineAdd`, or any combination containing one) is not linear;
see [`is_affine`](@ref). Every `LinearOperator` is linear; a combination is linear when all of
its parts are.

```jldoctest
julia> is_linear(DiagOp(rand(3)))
true

julia> is_linear(AffineAdd(DiagOp(rand(3)), rand(3)))
false
```
"""
is_linear(L::LinearOperator) = true

"""
	is_affine(A::AbstractOperator)

Returns true if `A` is affine: `A * x = Aₗ * x + d` for a linear `Aₗ` and a fixed displacement
`d` (see [`displacement`](@ref) and [`remove_displacement`](@ref)). Every linear operator is
affine; `AffineOperator` is the supertype of the operators that are affine by construction.

```jldoctest
julia> is_affine(AffineAdd(DiagOp(rand(3)), rand(3)))
true

julia> is_affine(Sin(3))
false
```
"""
is_affine(L::AffineOperator) = true

"""
	is_sliced(A)

Returns true if `A` is a sliced operator.
Operator `A` is sliced if it applies to only a subset of the input values.

```jldoctest
julia> is_sliced(DiagOp(rand(10)))
false

julia> is_sliced(DiagOp(rand(10)) * GetIndex((20,), 1:10))
true
```
"""
is_sliced(L) = false

"""
	get_slicing_expr(A)

Returns the slicing expression of `A`.
Operator `A` is sliced if it applies to only a subset of the input values.
The slicing expression is either a tuple of indices or a bit array that specifies the subset of input values that `A` applies to.
"""
get_slicing_expr(L) = is_null(L) ? nothing : Colon()

"""
	get_slicing_mask(A)

Returns the slicing mask of `A`.
Operator `A` is sliced if it applies to only a subset of the input values.
"""
get_slicing_mask(L) = error("cannot get slicing mask of operator of type $(typeof(L))")

"""
	remove_slicing(A)

Returns the operator `A` without slicing.
Operator `A` is sliced if it applies to only a subset of the input values.
"""
remove_slicing(L) = L

has_fast_opnorm(L) = false

"""
	displacement(A::AbstractOperator)

Returns the displacement of the operator: `A * 0`, as a scalar when all its entries are equal.

A linear operator (see [`is_linear`](@ref)) returns zero without being applied. Anything else is
applied to zeros, so a new operator needs a method only when it is affine or nonlinear and its
displacement is cheaper to state than to compute, as `AffineAdd` does.

```jldoctest
julia> A = AffineAdd(Eye(4),[1.;2.;3.;4.])
I+d  ℝ^4 -> ℝ^4

julia> displacement(A)
4-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0

```
"""
function displacement(S::AbstractOperator)
    is_linear(S) && return _zero_of(codomain_type(S))
    x = allocate_in_domain(S)
    fill!(x, 0)
    d = S * x
    # `d[1]`/iterating `d` directly would be scalar indexing on a GPU array, so the first
    # element comes back through a one-element host copy and the comparison is a reduction.
    # `d` itself (returned below) keeps its original storage type.
    v = _first_element(d)
    return _all_equal_to(d, v) ? v : d
end

_first_element(d::AbstractArray) = only(Array(@view vec(d)[1:1]))
# An `ArrayPartition` cannot be flattened when its blocks have incompatible shapes -- which is
# exactly what a per-frame operator with unequal sample counts produces -- so recurse instead.
_first_element(d::ArrayPartition) = _first_element(first(d.x))

_all_equal_to(d::AbstractArray, v) = all(==(v), d)
_all_equal_to(d::ArrayPartition, v) = all(b -> _all_equal_to(b, v), d.x)

# The first entry of `S * 0` for a linear `S`: of the first block's type for a block codomain.
_zero_of(T::Type) = zero(T)
_zero_of(T::Tuple) = _zero_of(first(T))

"""
	remove_displacement(A::AbstractOperator)

Removes the displacement of the operator.

"""
remove_displacement(A::AbstractOperator) = A

import Base: convert
function convert(::Type{T}, dom::Type, dim_in::Tuple, L::T) where {T <: AbstractOperator}
    domain_type(L) != dom &&
        error("cannot convert operator with domain $(domain_type(L)) to operator with domain $dom ")
    size(L, 1) != dim_in &&
        error("cannot convert operator with size $(size(L, 1)) to operator with domain $dim_in ")
    return L
end

"""
	can_be_combined(L::AbstractOperator, R::AbstractOperator) = false
	can_be_combined(L::AbstractOperator, M::AbstractOperator, R::AbstractOperator) = false

Returns whether the operators `L` and `R` can be merged when they are multiplied.

Examples:
```jldoctest; setup = :(using AbstractOperators)
julia> AbstractOperators.can_be_combined(DiagOp(rand(10)), FiniteDiff((10,)))
false

julia> AbstractOperators.can_be_combined(Eye(10), FiniteDiff((11,)))
true

julia> AbstractOperators.can_be_combined(FiniteDiff((10,)), Reshape(Eye(10), 2, 5))
false
```
"""
function can_be_combined(L, R)
    return _is_removable_eye(L) ||
        _is_removable_eye(R) ||
        is_null(L) ||
        (is_null(R) && is_linear(L))
end

"""
	_is_removable_eye(L)

Whether `L` is an identity that can simply be dropped from a composition.

`is_eye` alone is not enough: `Reshape(Eye(...), dims...)` is an identity on the *values* but not on the
*shape*, so removing it would leave the neighbouring operator with an input of the wrong number of
dimensions. Only a shape-preserving identity may be dropped.
"""
_is_removable_eye(L) = is_eye(L) && size(L, 1) == size(L, 2)
can_be_combined(L, M, R) = false

"""
	combine(L::AbstractOperator, R::AbstractOperator)
Returns the combined operator of `L` and `R`. The combined operator is defined as `L * R` where `L` and `R` are the operators to be combined.

Examples:
```jldoctest; setup = :(using AbstractOperators)
julia> AbstractOperators.combine(Eye(10), DiagOp(rand(10)))
╲  ℝ^10 -> ℝ^10
```
"""
function combine(L, R)
    if _is_removable_eye(L)
        return R
    elseif _is_removable_eye(R)
        return L
    elseif is_null(L)
        if size(R, 1) == size(R, 2) && domain_type(R) == codomain_type(R)
            return L
        else
            return Zeros(domain_type(R), size(R, 2), codomain_type(L), size(L, 1))
        end
    elseif is_null(R) && is_linear(L)
        if size(L, 1) == size(L, 2) && domain_type(L) == codomain_type(L)
            return R
        else
            return Zeros(domain_type(R), size(R, 2), codomain_type(L), size(L, 1))
        end
    else
        error("cannot combine operators")
    end
end

function combine(L, M, R)
    error("cannot combine operators")
end

"""
	has_optimized_normalop(L::AbstractOperator)

Returns whether the operator `L` has an optimized normal operator.
The normal operator is defined as `L' * L` where `L'` is the adjoint of `L`.
"""
has_optimized_normalop(L::AbstractOperator) = false

"""
	get_normal_op(L::AbstractOperator)

Returns the normal operator of the operator `L`. The normal operator is defined as `L' * L` where `L'` is the adjoint of `L`.
"""
function get_normal_op(L::AbstractOperator)
    return L' * L
end

"""
	LinearAlgebra.diag(A::AbstractOperator)

Returns the diagonal of `A`. If `A` is not diagonal, an error is thrown.

The diagonal is defined as the vector `d` such that `A * x = d .* x` for all `x` in the domain of `A`, where `.*` is the element-wise multiplication.
"""
LinearAlgebra.diag(L::AbstractOperator) = error("cannot get diagonal of operator of type $(typeof(L))")

"""
	LinearAlgebra.opnorm(A::AbstractOperator)

Returns the operator norm of `A`. The operator norm is defined as the maximum singular value of `A`.
It is computed with Lanczos (see `powerit`), unless the operator has a fast implementation.

The operator norm is defined as: `‖A‖ = sup_{x != 0} ‖A*x‖ / ‖x‖`.

Unless the operator has a fast implementation, this runs `powerit` with `maxit = 100` and
`rel_margin = 1e-6`, from a fixed pseudo-random start vector — so it is a deterministic function
of `A`, and, like every Krylov iteration, it approaches the norm from below.

Use `estimate_opnorm` to trade accuracy for time, to ask for a value that is guaranteed *not* to
fall below `‖A‖`, or to set the margin explicitly.
"""
function LinearAlgebra.opnorm(A::AbstractOperator)
    return powerit(A)
end

has_fast_opnorm(::AbstractOperator) = false

"""
	opnorm_bound(A::AbstractOperator)

A **certified upper bound** on `opnorm(A)`, in closed form, or `Inf` when none is known.

Every method must satisfy `opnorm_bound(A) >= opnorm(A)`. `Inf` propagates through the
combinator rules, so one unknown leaf makes the whole expression unknown rather than wrong.

`powerit` converges to `‖A‖` from below [1, §8.2], so it certifies only a *lower* bound; turning
it into an upper one needs a Kato–Temple gap estimate [2, Thm 4.6.1], which is not free. A
structural bound is the cheap certificate, and `estimate_opnorm` pairs the two into an interval.

The bounded quantity is the norm induced by the operator's **declared** adjoint — the same one
`powerit` measures — which for a non-unitary `DFT` normalization is not the textbook spectral
norm.

| operator      | bound                               | basis              |
|:--------------|:------------------------------------|:-------------------|
| `Compose`     | `prod` of the factors               | submultiplicativity [1, §2.3] |
| `VCAT`,`HCAT` | `sqrt` of the sum of squares        | `‖[A; B]x‖² = ‖Ax‖² + ‖Bx‖²`, Cauchy–Schwarz |
| `DCAT`        | `maximum` of the blocks             | blocks act on orthogonal subspaces [3, §2.1] |
| `Sum`         | `sum` of the terms                  | triangle inequality |
| `Scale`       | `abs(coeff)` times the bound        | exact              |
| `AffineAdd`   | the linear part, if `d == 0`        | see its method     |

A leaf with `has_fast_opnorm` contributes its exact norm. The result is always `Float64`,
whatever the operator's element type: the bound is one scalar, so the wider type is free, and a
method returning `Inf` beside one returning a `Float32` would make the recursion type-unstable.

## References

1. Golub, Van Loan, "Matrix Computations", 4th ed., Johns Hopkins (2013).
2. Parlett, "The Symmetric Eigenvalue Problem", SIAM Classics in Applied Mathematics 20 (1998).
3. Horn, Johnson, "Topics in Matrix Analysis", Cambridge (1991).

See also: `estimate_opnorm`, `has_fast_opnorm`, `powerit`.
"""
opnorm_bound(A::AbstractOperator) = has_fast_opnorm(A) ? Float64(LinearAlgebra.opnorm(A)) : Inf

"""
	estimate_opnorm(A::AbstractOperator)

Estimates the operator norm of `A`. The operator norm is defined as the maximum singular value of `A`.
It is computed by an iteration on `AᴴA` (Lanczos or the power method, see `method`) unless the operator
has a fast implementation.

The operator norm is defined as: `‖A‖ = sup_{x != 0} ‖A*x‖ / ‖x‖`.

## Keyword arguments

- `rel_margin = 0.01`: how far from `‖A‖` the result may be. See "What is returned" below for
  what it does and does not guarantee.
- `side = :upper`: `:upper` never returns a value below `‖A‖`, which is what a Lipschitz
  constant needs; `:accurate` returns the closest value instead, which is what a rescaling
  needs. See "Which side to ask for".
- `maxit`: iteration cap, 100 by default. Without a closed-form bound the iteration runs the
  number of steps `failure_probability` needs instead, and a smaller `maxit` raises that
  probability, with a warning.
- `rng`: start vector source, a **fixed-seed** generator by default (see `powerit`), so
  repeated calls on the same operator return the same number.
- `threaded = true`: whether the iteration's own vector updates may thread. The applications
  of `A` follow `A`'s own threading.
- `method = :auto`: the iteration, `:lanczos`, `:power` or `:auto`. See "Which iteration runs".
- `failure_probability = 1e-3`: without a closed-form bound, the probability over the start
  vector that the `:upper` value falls below `‖A‖`. `nothing` gives up that guarantee for the
  residual estimate `sqrt(θₖ + ‖rₖ‖)`, which is exact when the iteration has found the top
  eigenvalue and below `‖A‖` when it has not. See "Without a certificate".

## What is returned

`powerit` gives a certified lower bound `L`; [`opnorm_bound`](@ref) gives a certified upper bound
`U`, or `Inf`. This combines them:

| case | result | guarantee |
|:-----|:-------|:----------|
| `has_fast_opnorm(A)` | `opnorm(A)` | exact, no iteration |
| `side = :upper`, `U` finite | `U` | `U ≥ ‖A‖`, certified |
| `side = :upper`, `U = Inf` | `sqrt(θₖ) (1 + rel_margin)` | `≥ ‖A‖` with probability `1 - failure_probability`, see below |
| `side = :upper`, `U = Inf`, `failure_probability = nothing` | `sqrt(θₖ + ‖rₖ‖)` | none, see below |
| `side = :accurate` | `L` | `≤ ‖A‖` |

`rel_margin` is a promise about the value returned, and it is kept in every row. A certificate is
never known to be within the margin on its own — `U ≤ L (1 + rel_margin)` is the only computable
statement of that — so the iteration runs until it holds, and `U` is loose exactly when it cannot
be made to. If `maxit` runs out first, `U` is returned with a warning naming the achieved slack:
still safe, merely loose, and loose costs convergence rate while low costs convergence.

## Without a certificate

With `U = Inf` no computable quantity bounds `‖A‖` from above: `θ + ‖r‖` bounds *some* eigenvalue
near `θ` [2, Thm 4.5.1], not necessarily the largest, and an iteration that has not yet seen the
top eigenvector cannot tell. One isolated top singular value over a dense band of the rest, which a
random start vector meets with weight `1/√n`, makes both iterations converge on the band and stop:
with the band at 95% of `λmax` and `n = 10⁶`, the residual bound came out 1.9% (Lanczos) and 2.5%
(power method) below `‖A‖` at a 1% margin.

What can be bounded is the probability. From a start vector uniform on the sphere, the top
estimate `θₖ` after `k` steps falls below `(1 - ε) λmax` with probability at most
`1.648 √n e^{-√ε (2k - 1)}` for Lanczos and `0.824 √n (1 - ε)^{k - 1/2}` for the power method [3],
`n` the real dimension of the domain. With `1 - ε = (1 + rel_margin)^{-2}`, running the `k` that
brings this to `failure_probability` and returning `sqrt(θₖ) (1 + rel_margin)` gives a value that is
at or above `‖A‖` with that probability, and, since `θₖ ≤ λmax`, never above
`‖A‖ (1 + rel_margin)`. The step count depends only on `n`, the margin and the probability:

| `rel_margin` | Lanczos, `n = 10⁴ … 10⁷` | power method |
|:-------------|:--------------------------|:-------------|
| `0.1` | 15-20 | 60-79 |
| `0.01` | 44-56 | 570-743 |
| `0.001` | 135-174 | 5663-7391 |

(at `failure_probability = 1e-3`). The probability is over start vectors: the default start vector
comes from a fixed seed, so an operator built against that vector can still defeat it.

These counts are for the worst spectrum of each size. A spectrum whose top eigenvalue stands well
clear of the rest is resolved in a handful of steps, and a caller who knows its operators are of
that kind can pass `failure_probability = nothing`: the iteration then stops on the residual test
`‖rₖ‖ / (2θₖ) ≤ rel_margin`, as with a certificate, and returns `sqrt(θₖ + ‖rₖ‖)`, at most
`‖A‖ (1 + rel_margin)` and at or above `‖A‖` whenever the iteration has converged to the top
eigenvalue. Nothing checks that it has: on the isolated-eigenvalue example above it is the value
that came out 1.9% low.

## Which iteration runs

`:auto` and `:lanczos` run Lanczos; `:power` runs the power method. Lanczos builds the Krylov space
the power method moves in and takes its best vector, so it never needs more applications of `AᴴA`,
and on a slow-gapped, clustered or flat top of the spectrum it needs several times fewer
[2, ch. 13]; without a certificate the guarantee above costs it 4 to 40 times fewer steps.

Without reorthogonalisation the Lanczos residual is the estimate `βₖ |sₖ|`, exact while the basis
stays orthogonal; past convergence, once the basis loses orthogonality (Paige; the Ritz values stay
inside the spectrum), it can fall far below the true residual, so the residual returned is floored
at `k ε θ`, the order of the rounding after `k` steps. The power method forms its residual as a
vector at every step. The residual decides when the iteration stops, and enters the value returned
only with `failure_probability = nothing`, where the floor keeps it from coming out too small.

## Which side to ask for

`:upper` is right for a **Lipschitz constant**: Beck and Teboulle [1, §4] require `L ≥ L(∇f)`,
and with a fixed step `γ = 1/Lf` nothing corrects a low value. It is wrong for rescaling, penalty
selection or seeding a backtracking search, where accuracy matters and neither direction is
unsafe — pass `:accurate` there, and never for a step size.

## References

1. Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
   Problems", SIAM J. Imaging Sciences 2(1), 183-202 (2009).
2. Parlett, "The Symmetric Eigenvalue Problem", SIAM Classics in Applied Mathematics 20 (1998).
3. Kuczyński, Woźniakowski, "Estimating the largest eigenvalue by the power and Lanczos algorithms
   with a random start", SIAM J. Matrix Anal. Appl. 13(4), 1094-1122 (1992).

See also: [`opnorm_bound`](@ref), `powerit`, `has_fast_opnorm`.
"""
function estimate_opnorm(
        A::AbstractOperator;
        rel_margin = 0.01,
        side::Symbol = :upper,
        maxit = nothing,
        rng = _powerit_rng(),
        threaded::Bool = true,
        method::Symbol = :auto,
        failure_probability = 1.0e-3,
    )
    side in (:upper, :accurate) ||
        throw(ArgumentError("`side` must be `:upper` or `:accurate`, got $(repr(side))"))
    has_fast_opnorm(A) && return opnorm(A)

    # `:accurate` wants the closest value, and that is the power iteration's own lower bound: it
    # falls short of `‖A‖` by the square of the iterate's angle error, whereas `opnorm_bound` is a
    # structural over-estimate that no amount of iteration improves. Since the bound can never
    # come out below the iterate, it has nothing to contribute here and is not even computed.
    side === :accurate &&
        return first(_powerit(A; maxit = something(maxit, 100), rel_margin, rng, upper = Inf, threaded, method))

    upper = opnorm_bound(A)
    isfinite(upper) || return _probable_upper_bound(A; rel_margin, failure_probability, maxit, rng, threaded, method)
    # `rel_margin` is a promise about the value returned, so it is checked against the value
    # returned. A certificate is only known to be within the margin once the iteration has
    # climbed to meet it, which is what `_powerit` is asked to do when `upper` is finite.
    maxit = something(maxit, 100)
    lower, θ, resid = _powerit(A; maxit, rel_margin, rng, upper, threaded, method)

    if upper > lower * (1 + rel_margin) && lower > 0
        @warn "estimate_opnorm: the closed-form bound is looser than the requested margin" achieved =
            upper / lower - 1 rel_margin maxit maxlog = 1
    end
    # `:upper` promises a value at or above `‖A‖`. Narrowing the `Float64` bound to a
    # `Float32` iterate rounds to nearest, which can land below the bound and so below `‖A‖`,
    # so it is rounded up instead. Nor is the value ever less than the iteration certified:
    # a bound that is only structurally justified is no proof against a `lower` that came
    # out above it.
    U = oftype(lower, upper)
    U < upper && (U = nextfloat(U))
    return max(U, lower)
end

# `side = :upper` without a closed-form bound: `sqrt(θₖ) (1 + rel_margin)` after the number of steps
# `k` that makes it fall below `‖A‖` with probability at most `failure_probability` over the start
# vector. `θₖ` never exceeds `λmax(AᴴA)`, so the value never exceeds `‖A‖ (1 + rel_margin)`. With
# `failure_probability = nothing`, the residual estimate after the residual test instead.
function _probable_upper_bound(A; rel_margin, failure_probability, maxit, rng, threaded, method)
    rel_margin > 0 || throw(ArgumentError("`rel_margin` must be positive without a closed-form bound, got $rel_margin"))
    if failure_probability === nothing
        _, θ, resid = _powerit(A; maxit = something(maxit, 100), rel_margin, rng, upper = Inf, threaded, method)
        return sqrt(θ + resid)
    end
    0 < failure_probability < 1 ||
        throw(ArgumentError("`failure_probability` must lie in (0, 1), got $failure_probability"))
    m = _norm_method(method)
    n = _real_dimension(domain_type(A), size(A, 2))
    ε = 1 - 1 / (1 + rel_margin)^2
    k = _guaranteed_steps(m, n, ε, failure_probability)
    if maxit !== nothing && maxit < k
        @warn "estimate_opnorm: `maxit` is below the steps the failure probability needs" maxit needed = k achieved =
            _failure_probability(m, n, ε, maxit) failure_probability maxlog = 1
        k = maxit
    end
    _, θ, _ = _powerit(A; maxit = k, rel_margin, rng, upper = Inf, threaded, method, steps = k)
    return sqrt(θ) * (1 + oftype(θ, rel_margin))
end

# Kuczyński and Woźniakowski [3, Thm 4.2 and Thm 3.1]: from a start vector uniform on the sphere of
# a real `n`-dimensional space, after `k` steps the estimate `θₖ` of the top eigenvalue of a
# positive semidefinite matrix falls below `(1 - ε) λmax` with probability at most
# `1.648 √n e^{-√ε (2k - 1)}` for Lanczos and `0.824 √n (1 - ε)^{k - 1/2}` for the power method.
_failure_probability(m::Symbol, n, ε, k) =
    m === :power ? 0.824 * sqrt(n) * (1 - ε)^(k - 0.5) : 1.648 * sqrt(n) * exp(-sqrt(ε) * (2k - 1))
_guaranteed_steps(m::Symbol, n, ε, δ) = max(
    1, m === :power ? ceil(Int, log(δ / (0.824 * sqrt(n))) / log(1 - ε) + 0.5) :
        ceil(Int, (log(1.648 * sqrt(n) / δ) / sqrt(ε) + 1) / 2),
)

# The dimension of the domain as a real space: a complex entry is two real ones, and a complex
# Gaussian start vector is a real Gaussian one there.
_real_dimension(T::Type, sz::Tuple{Vararg{Int}}) = prod(sz; init = 1) * (T <: Complex ? 2 : 1)
_real_dimension(T::Tuple, sz::Tuple) = sum(map(_real_dimension, T, sz))

# A fresh, fixed-seed generator per call, so the start vector does not depend on the global
# RNG's state and therefore not on what the caller happened to draw before.
_powerit_rng() = Random.Xoshiro(0x5eed)

"""
	powerit(A::AbstractOperator; maxit, rel_margin, rng, threaded)

A **lower** bound on `‖A‖` from an iteration on `AᴴA`: Lanczos by default (`method = :auto`
or `:lanczos`), or the power method (`method = :power`). See `estimate_opnorm`, "Which iteration
runs".

The iterates approach `‖A‖` from below and never cross it, so this is never safe as a step-size
denominator; `estimate_opnorm` is, since it pairs this with a certified upper bound.

`rel_margin` is an error target, not a progress test. The keyword it replaced, `tol`, compared
two successive iterates — how fast the iteration moves, not how far it has left. Measured on a
128²×8 operator, it stopped at the same value for `maxit = 40` and `maxit = 80`, 0.84% below the
truth.

The start vector is drawn from `rng`, a **fixed-seed** generator by default. That is not only for
reproducible tests: convergence is linear in `|λ₂/λ₁|` [1, §8.2.1], so with a near-degenerate top
of the spectrum `maxit` is usually exhausted and the start vector leaks into the *result*. From
the global RNG the same operator gave a 6.5e-4 relative spread over six calls. It is drawn on the
host and copied into `A`'s domain storage, so a device operator gets the same start vector.

`threaded = false` keeps the iteration's vector updates on one thread.

## References

1. Golub, Van Loan, "Matrix Computations", 4th ed., Johns Hopkins (2013).
"""
function powerit(
        A::AbstractOperator; maxit = 100, rel_margin = 1.0e-6, rng = _powerit_rng(), threaded::Bool = true,
        method::Symbol = :auto,
    )
    return first(_powerit(A; maxit, rel_margin, rng, upper = Inf, threaded, method))
end

"""
	_powerit(A; maxit, rel_margin, rng, upper, threaded, method) -> (lower, θ, resid)

One iteration on `B = AᴴA`, Lanczos or the power method by `method` (see `estimate_opnorm`,
"Which iteration runs"), reporting what its callers need rather than just a number.

The power method:

- `lower = sqrt(‖Bx‖)` for the final unit iterate `x`, a certified lower bound on `‖A‖`. `‖Bx‖`
  beats the Rayleigh quotient here because `θ ≤ ‖Bx‖ ≤ λmax` for positive semidefinite `B`.
- `θ = xᴴBx`, the Rayleigh quotient.
- `resid = ‖Bx - θx‖`, formed as the vector it is rather than as `sqrt(‖Bx‖² - θ²)`. The residual
  is orthogonal to `x` [1, §4.3], so the two agree in exact arithmetic, but the subtraction is a
  difference of nearly equal squares and loses half the significant digits: in `Float32` it cannot
  resolve a residual below about `3e-4 ‖Bx‖`, and near convergence it goes negative and reads as
  exact convergence. One extra vector and one extra pass buy those digits back.

Lanczos [1, ch. 13]:

- `θ`, the largest eigenvalue of the tridiagonal `Tₖ`, and `lower = sqrt(θ)`. The Ritz values
  interlace the spectrum of `B`, so `θ ≤ λmax`. Without reorthogonalisation that holds to
  rounding (Paige).
- `resid = max(βₖ |sₖ|, k ε θ)` for the eigenvector `s` of `Tₖ` and the unit roundoff `ε` of the
  element type. `βₖ |sₖ|` is the residual of the Ritz pair while the basis is orthogonal; past
  convergence, once orthogonality is lost, it can fall far below the true residual, and `k ε θ`,
  the order of the rounding Paige's analysis allows after `k` steps, is its floor. The loop's own
  stop test uses the estimate unfloored.

The loop stops on `resid / (2θ) ≤ rel_margin` — the factor 2 is the square root between `λ` and
`‖A‖`, and dropping it makes the test twice as strict as asked — or, with a finite `upper`, as
soon as `upper ≤ lower (1 + rel_margin)`. The second test is what lets a caller returning `upper`
keep its promise about the margin: the certificate itself does not improve with iteration, but
whether it sits within the margin can only be established by raising `lower` to meet it. Either
test ending the loop ends it, since neither the certificate nor the iterate has anything left to
gain from the other's criterion.

## References

1. Parlett, "The Symmetric Eigenvalue Problem", SIAM Classics in Applied Mathematics 20 (1998).
"""
function _powerit(
        A::AbstractOperator; maxit, rel_margin, rng, upper, threaded::Bool = true, method::Symbol = :auto,
        steps::Union{Nothing, Int} = nothing,
    )
    _norm_method(method) === :power && return _power_iteration(A; maxit, rel_margin, rng, upper, threaded, steps)
    return _lanczos(A; maxit, rel_margin, rng, upper, threaded, steps)
end

const _NORM_METHODS = (:auto, :lanczos, :power)

# The iteration `method` names; `:auto` is Lanczos.
function _norm_method(method::Symbol)
    method in _NORM_METHODS ||
        throw(ArgumentError("`method` must be one of $(join(map(repr, _NORM_METHODS), ", ")), got $(repr(method))"))
    return method === :auto ? :lanczos : method
end

function _power_iteration(A::AbstractOperator; maxit, rel_margin, rng, upper, threaded::Bool, steps = nothing)
    AHA = A' * A
    x = allocate_in_domain(A)
    y = similar(x)
    r = similar(x)
    copyto!(x, randn(rng, eltype(x), size(x)))
    normalize!(x)
    R = real(eltype(x))
    nrm = zero(R)
    θ = zero(R)
    resid = R(Inf)

    for _ in 1:maxit
        mul!(y, AHA, x)
        nrm = norm(y)
        # A null operator: every bound is zero and dividing by `nrm` below would not be defined.
        nrm == 0 && return (zero(R), zero(R), zero(R))
        θ = real(dot(x, y))
        if threaded
            @.. thread = true r = y - θ * x
        else
            @.. r = y - θ * x
        end
        resid = norm(r)
        if steps === nothing
            θ > 0 && resid / (2θ) <= rel_margin && break
            isfinite(upper) && upper <= sqrt(nrm) * (1 + rel_margin) && break
        end
        if threaded
            @.. thread = true x = y / nrm
        else
            @.. x = y / nrm
        end
    end

    return (sqrt(nrm), θ, resid)
end

# Lanczos on `B = AᴴA` without reorthogonalisation, from the same start vector as the power
# iteration. The tridiagonal coefficients are kept on the host in `Float64`; the vectors stay in
# `A`'s domain storage.
function _lanczos(A::AbstractOperator; maxit, rel_margin, rng, upper, threaded::Bool, steps = nothing)
    AHA = A' * A
    q = allocate_in_domain(A)
    copyto!(q, randn(rng, eltype(q), size(q)))
    normalize!(q)
    q_prev = zero(q)
    w = similar(q)
    R = real(eltype(q))
    α = Float64[]
    β = Float64[]
    θ = 0.0
    s = Float64[]
    resid = Inf

    for _ in 1:maxit
        mul!(w, AHA, q)
        a = real(dot(q, w))
        b_prev = R(isempty(β) ? 0.0 : β[end])
        if threaded
            @.. thread = true w = w - R(a) * q - b_prev * q_prev
        else
            @.. w = w - R(a) * q - b_prev * q_prev
        end
        b = Float64(norm(w))
        push!(α, a)
        push!(β, b)
        θ, s = _top_ritz_pair(α, β)
        # A null operator: every bound is zero.
        θ <= 0 && b == 0 && return (zero(R), zero(R), zero(R))
        resid = b * abs(s[end])
        # `β = 0`: the Krylov space is invariant and `θ` is an eigenvalue of `B`.
        b == 0 && break
        if steps === nothing
            θ > 0 && resid / (2θ) <= rel_margin && break
            isfinite(upper) && upper <= sqrt(θ) * (1 + rel_margin) && break
        end
        q, q_prev = q_prev, q
        if threaded
            @.. thread = true q = w / R(b)
        else
            @.. q = w / R(b)
        end
    end
    return (R(sqrt(θ)), R(θ), R(max(resid, length(α) * eps(R) * θ)))
end

# The largest eigenvalue of the Lanczos tridiagonal, by bisection, and its eigenvector, by inverse
# iteration: O(k) each, where a full eigendecomposition per step would make a long run cubic.
# `eigen(::SymTridiagonal)` also goes through `stegr`, which fails on the nearly equal eigenvalues
# a run past convergence produces.
function _top_ritz_pair(α::Vector{Float64}, β::Vector{Float64})
    k = length(α)
    d, e = copy(α), β[1:(k - 1)]
    w, iblock, isplit = LinearAlgebra.LAPACK.stebz!('I', 'E', 0.0, 0.0, k, k, 0.0, d, e)
    s = LinearAlgebra.LAPACK.stein!(d, e, w, iblock, isplit)
    return w[1], vec(s)
end

#printing
function Base.show(io::IO, L::AbstractOperator)
    return print(io, fun_name(L) * storage_display_string(L) * " " * fun_space(L))
end

function fun_space(L::AbstractOperator)
    dom = fun_dom(L, 2)
    codom = fun_dom(L, 1)
    return dom * "->" * codom
end

function fun_dom(L::AbstractOperator, n::Int)
    dm = n == 2 ? domain_type(L) : codomain_type(L)
    sz = size(L, n)
    return string_dom(dm, sz)
end

function string_dom(dm::Type, sz::Tuple)
    dm_st = dm <: Complex ? " ℂ" : " ℝ"
    sz_st = length(sz) == 1 ? "$(sz[1]) " : "$sz "
    return dm_st * "^" * sz_st
end

function string_dom(dm::Tuple, sz::Tuple)
    s = string_dom.(dm, sz)
    return length(s) > 3 ? s[1] * "..." * s[end] : *(s...)
end

"""
    copy_operator(op::AbstractOperator; storage_type=nothing, threaded=nothing)

Create a copy of `op` suitable for parallel use. **Always returns a new object**; use
[`adapt_operator`](@ref) when sharing `op` is acceptable provided it meets the constraints.

- Immutable fields (operator arrays, type params) are **shared** (no copy).
- Mutable buffer fields are **deep-copied**.
- `storage_type`: if provided (e.g., `CuArray`), convert buffer arrays to that storage.
- `threaded`: if provided (`true`/`false`), request that threading state for operators that
  support it; `nothing` (the default) preserves whatever the operator already has.

Note these two are **constraint** arguments, which is why they accept `nothing` while the
operator *constructors* take a plain `threaded::Bool`. `nothing` here means "no constraint
on this axis", exactly as it does for `storage_type` — without it a plain `copy_operator(op)`
could not preserve an explicitly serial operator, since it would re-derive threading from
the policy. As everywhere else, a `threaded = true` request is a permission the policy may
still decline; `threaded = false` is honoured absolutely.

The "return `op` itself when it is already thread-safe" short-circuit that used to live
here now lives in `adapt_operator`, so that the two functions have crisp contracts: one
always copies, the other never copies needlessly.
"""
function copy_operator(op::AbstractOperator; storage_type = nothing, threaded = nothing)
    return _copy_operator_impl(op; storage_type = _normalize_storage_request(storage_type), threaded)
end

# `_copy_operator_impl` methods uniformly build parameterized types as `storage_type{T}`,
# so the request must arrive as a bare wrapper (`Array`, `CuArray`). Callers naturally
# write either form, and `Array{Float64}{Float64}` is a confusing `MethodError` rather than
# a helpful one -- so normalize here, once, instead of in every impl.
_normalize_storage_request(::Nothing) = nothing
_normalize_storage_request(S::Type{<:AbstractArray}) = _array_wrapper_type(S)

# Fallback. `deepcopy` reproduces the operator exactly as it is, so it can only answer a
# request it does not have to change anything for. Two cases qualify:
#
#   * `storage_type === nothing` — nothing to convert.
#   * `threaded` given but `supports_threading(op) == false` — the operator has no threaded
#     path, so the request is vacuous. This is the same rule `_satisfies_constraints` uses,
#     and it is what lets a threaded batch operator wrap an FFTW/DSP operator: those are
#     never threaded themselves, so `threaded = false` asks nothing of them.
#
# Anything else would return an operator that does not meet the caller's constraints, so
# refuse instead, naming the type that needs a `_copy_operator_impl` method.
function _copy_operator_impl(
        op::T; storage_type = nothing, threaded = nothing
    ) where {T <: AbstractOperator}
    if storage_type === nothing && (threaded === nothing || !supports_threading(op))
        return deepcopy(op)
    end
    unmet = storage_type === nothing ? "threaded" : "storage_type"
    return throw(
        ArgumentError(
            "copy_operator cannot honour `$(unmet)` for $(T): no _copy_operator_impl " *
                "method is defined for it, and the deepcopy fallback would silently " *
                "ignore the request. Define " *
                "AbstractOperators._copy_operator_impl(::$(T); storage_type, threaded)."
        ),
    )
end

# Helper: convert a buffer array to the target storage type
function _convert_buffer(buf::AbstractArray, ::Nothing)
    return similar(buf)  # same type, new allocation
end
function _convert_buffer(buf::AbstractArray{T}, storage_type::Type) where {T}
    return similar(storage_type{T}, size(buf))
end

_should_thread(::Number) = false
_should_thread(d::AbstractArray) = length(d) >= THRESHOLD_MEMORY_BOUND && Threads.nthreads() > 1
_should_thread(S::Type{<:AbstractArray}) = Threads.nthreads() > 1 && _is_cpu_storage(S)

"""
	_should_thread(op::AbstractOperator)

Whether a *batch* loop over `op` should thread.

Requires more than one Julia thread, CPU storage, and the wrapped operator to carry at
least `MIN_BATCH_WORK_FOR_PARALLEL` elements per call -- otherwise a batch operator would
thread at any size, including a four-element one, paying the full `@budgeted_threads`
setup to parallelise microseconds of work.

The batch *count* is not known here (it is decided by the caller), so this is deliberately
only the per-item half of the condition.
"""
# `storage_type`: the decision for a copy of `op` moved there (`nothing`: unchanged).
function _should_thread(op::AbstractOperator, storage_type = nothing)
    Threads.nthreads() > 1 || return false
    if storage_type === nothing
        _is_cpu_storage(_policy_storage(domain_array_type(op))) || return false
    else
        _is_cpu_domain(storage_type) || return false
    end
    return _total_elements(size(op, 2)) >= MIN_BATCH_WORK_FOR_PARALLEL
end

"""
	_should_thread(op::AbstractOperator, batch_length::Integer)

Whether a batch loop of `batch_length` calls to `op` should thread.

Besides the per-item condition above, a batch with fewer items than threads stays serial
when `op` threads itself: the loop could keep only `batch_length` threads busy, while each
call can use all of them. For two frames of a 2-D wavelet transform (db2, 3 levels,
ComplexF32, 128² to 512², one EPYC 7763 NUMA domain), threading inside each frame took
0.39-0.63 of the time of threading across the frames at 8 threads, and 0.26-0.57 at 16.
"""
function _should_thread(op::AbstractOperator, batch_length::Integer)
    _should_thread(op) || return false
    return batch_length >= Threads.nthreads() || !is_threaded(op)
end

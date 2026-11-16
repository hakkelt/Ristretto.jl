export HCAT

"""
	HCAT(A::AbstractOperator...)

Shorthand constructors:

	[A1 A2 ...]
	hcat(A...)

Horizontally concatenate `AbstractOperator`s. Notice that all the operators must share the same codomain dimensions and type, e.g. `size(A1,1) == size(A2,1)` and `codomain_type(A1) == codomain_type(A2)`.

```jldoctest
julia> HCAT(Eye(10),FiniteDiff((20,))[1:10])
[I,↓*δx]  ℝ^10  ℝ^20 -> ℝ^10

julia> H = [Eye(10) DiagOp(2*ones(10))]
[I,╲]  ℝ^10  ℝ^10 -> ℝ^10

julia> hcat(H,FiniteDiff((11,)))
HCAT  ℝ^10  ℝ^10  ℝ^11 -> ℝ^10

julia> # To evaluate `HCAT` operators multiply them with a `Tuple` of `AbstractArray` of the correct dimensions and type.

julia> using RecursiveArrayTools

julia> H*ArrayPartition(ones(10),ones(10))
10-element Vector{Float64}:
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
	
```
"""
struct HCAT{
        N, # number of AbstractOperator
        L <: NTuple{N, AbstractOperator},
        P <: Tuple,
        C <: AbstractArray,
        DS <: AbstractArray,  # domain storage type (fixed at construction)
        Th,                   # thread the adjoint block loop?
        LP,                   # type of A_par (== L when Th is false: no separate copy)
    } <: AbstractOperator
    A::L     # tuple of AbstractOperators, unmodified -- used by the (always-serial) forward
    # direction and by every property/introspection method (_children, is_threaded,
    # domain_type, size, ...), so a block's own threading state is never silently lost.
    idxs::P  # indices
    # H = HCAT(Eye(n),HCAT(Eye(n),Eye(n))) has H.idxs = (1,2,3)
    # `AbstractOperators` are flatten
    # H = HCAT(Eye(n),Compose(MatrixOp(randn(n,n)),HCAT(Eye(n),Eye(n))))
    # has H.idxs = (1,(2,3))
    # `AbstractOperators` are stack
    buf::C   # buffer memory
    A_par::LP  # `A` with every block forced `threaded = false`; used only by the threaded
    # adjoint block loop, to avoid nesting each block's own threading inside that loop.
    # Blocks that share a non-thread-safe operator are copied; see `_block_loop_operators`.
    # Aliases `A` (LP === L, no extra copy) whenever the adjoint loop itself is not threaded.
    function HCAT(A::L, idxs::P, buf::C; threaded::Bool = true) where {N, L <: NTuple{N, AbstractOperator}, P <: Tuple, C}
        if any([size(A[1], 1) != size(a, 1) for a in A])
            throw(DimensionMismatch("operators must have the same codomain dimension!"))
        end
        if any([codomain_type(A[1]) != codomain_type(a) for a in A])
            throw(error("operators must all share the same codomain_type!"))
        end
        DS = _compute_hcat_ds(A, idxs)
        # Only the *adjoint* direction is block-parallel: it writes disjoint domain blocks.
        # The forward direction accumulates every block into one shared `y`, so it stays
        # serial -- the mirror image of VCAT.
        # See the note in DCAT: `A` must not be reassigned while a closure captures it,
        # or it gets boxed to `Any` and the whole constructor goes through runtime dispatch.
        th = _resolve_threaded(threaded) do
            default_block_threaded(HCAT, A)
        end
        A_par = th ? _block_loop_operators(A) : A
        return new{N, L, P, C, DS, th, typeof(A_par)}(A, idxs, buf, A_par)
    end
end

function _compute_hcat_ds(A, idxs)
    ds_list = [d <: ArrayPartition ? [d.parameters[2].types...] : d for d in domain_array_type.(A)]
    domain = vcat(ds_list...)
    p = vcat([[idx...] for idx in idxs]...)
    invpermute!(domain, p)
    T = promote_type(map(_storage_eltype, domain)...)
    return ArrayPartition{T, Tuple{domain...}}
end

function HCAT(A::Vararg{AbstractOperator})
    if any((<:).(typeof.(A), HCAT)) #there are HCATs in A
        AA = ()
        for a in A
            if typeof(a) <: HCAT # flatten
                AA = (AA..., a.A...)
            else                 # stack
                AA = (AA..., a)
            end
        end
        # use buffer from HCAT in A
        buf = A[findfirst((<:).(typeof.(A), HCAT))::Int].buf
    else
        AA = A
        # generate buffer
        buf = allocate_in_codomain(AA[1])
    end

    return HCAT(AA, buf)
end

# Count actual domain slots from an HCAT's P (idxs) type.
# Each entry in P is either an Int (1 slot) or a NTuple{n,Int} (n slots).
_count_hcat_ndoms(::Type{<:Tuple{}}) = 0
@generated function _count_hcat_ndoms(::Type{P}) where {P <: Tuple}
    K = 0
    for i in 1:fieldcount(P)
        Pi = fieldtype(P, i)
        K += Pi <: Integer ? 1 : fieldcount(Pi)
    end
    return :($K)
end

# compile-time domain ndoms for HCAT's sub-operators:
# use the index-tuple type P (not N which only counts sub-operators) so that
# sub-operators with multi-component domains are accounted for correctly.
_ndoms_from_type(::Type{<:HCAT{N, L, P}}, dim::Int) where {N, L, P} =
    dim == 2 ? _count_hcat_ndoms(P) : 1

function HCAT(AA::NTuple{N, AbstractOperator}, buf; threaded::Bool = true) where {N}
    return _hcat_impl(AA, buf, threaded)
end

@generated function _hcat_impl(AA::NTuple{N, AbstractOperator}, buf::C, threaded) where {N, C}
    N == 1 && return :(AA[1])
    # Build idxs at compile time using operator element types
    K = 0
    idx_exprs = []
    for i in 1:N
        nd = _ndoms_from_type(fieldtype(AA, i), 2)
        if nd == 1
            K += 1
            push!(idx_exprs, K)
        else
            K0 = K
            push!(idx_exprs, ntuple(j -> K0 + j, nd))
            K += nd
        end
    end
    idxs_literal = Expr(:tuple, idx_exprs...)
    return :(HCAT(AA, $idxs_literal, buf; threaded))
end

HCAT(A::AbstractOperator) = A

# Mappings
# `ArrayPartition(b...)` only wraps the tuple, so checking through it copies nothing.
function mul!(y::AbstractArray, H::HCAT, b::ArrayPartition)
    check(y, H, b)
    return _hcat_mul!(y, H, b.x)
end

function mul!(y::AbstractArray, H::HCAT, b::Tuple)
    check(y, H, ArrayPartition(b...))
    return _hcat_mul!(y, H, b)
end

function _hcat_mul!(y, H::HCAT, b::Tuple)
    if _hcat_has_natural_idxs(H)
        return _mul_hcat_natural!(y, H, b)
    end
    return _mul_hcat_indexed!(y, H, b)
end

function mul!(y::AbstractArray, H::HCAT, b::ArrayPartition, α::Number, β::Number)
    check(y, H, b)
    iszero(β) && isone(α) && return mul!(y, H, b.x)
    if _hcat_has_natural_idxs(H)
        return _mul_hcat_natural!(y, H, b.x, α, β, Val(false))
    end
    return _mul_hcat_indexed!(y, H, b.x, α, β, Val(false))
end

function mul!(y::ArrayPartition, A::AdjointOperator{<:HCAT}, b::AbstractArray)
    check(y, A, b)
    _hcat_adj_mul!(y.x, A, b)
    return y
end

function mul!(y::Tuple, A::AdjointOperator{<:HCAT}, b::AbstractArray)
    check(ArrayPartition(y...), A, b)
    return _hcat_adj_mul!(y, A, b)
end

# Every block owns its part of `y`, so each accumulates there by itself.
@generated function mul!(
        y::ArrayPartition, A::AdjointOperator{<:HCAT{N, L, P}}, b::AbstractArray, α::Number, β::Number
    ) where {N, L, P}
    function output_expr(i)
        Pi = fieldtype(P, i)
        Pi <: Integer && return :(y.x[H.idxs[$i]])
        return :(ArrayPartition($((:(y.x[H.idxs[$i][$j]]) for j in 1:fieldcount(Pi))...)))
    end
    ex = :(check(y, A, b); H = A.A)
    for i in 1:N
        ex = :($ex; mul!($(output_expr(i)), H.A[$i]', b, α, β))
    end
    return :($ex; return y)
end

@generated function _hcat_adj_mul!(
        y::Tuple, A::AdjointOperator{<:HCAT{N, L, P, C, DS, false}}, b::AbstractArray
    ) where {N, L, P, C, DS}
    K = 0
    function output_natural_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            K += 1
            return :(y[$K])
        else
            n = fieldcount(Pi)
            parts = [:(y[$(K + j)]) for j in 1:n]
            K += n
            return :(ArrayPartition($(parts...)))
        end
    end

    ex = :(H = A.A)
    function output_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            return :(y[H.idxs[$i]])
        else
            n = fieldcount(Pi)
            parts = [:(y[H.idxs[$i][$j]]) for j in 1:n]
            return :(ArrayPartition($(parts...)))
        end
    end

    ex_natural = ex
    for i in 1:N
        ex_natural = :($ex_natural; mul!($(output_natural_expr(i)), H.A[$i]', b))
    end
    ex_natural = :($ex_natural; return y)

    ex_indexed = ex
    for i in 1:N
        ex_indexed = :($ex_indexed; mul!($(output_expr(i)), H.A[$i]', b))
    end
    ex_indexed = :($ex_indexed; return y)

    return :(_hcat_has_natural_idxs(A.A) ? ($ex_natural) : ($ex_indexed))
end

# Threaded adjoint. Same structure as the serial method above -- including its runtime
# choice between the natural-order and indexed expansions -- but with the per-block call
# lifted into a `Val(i)`-selected helper so the parallel loop can live in a plain function
# (a `@generated` body must be pure, and the threading macros expand to closures).
#
# Reads `H.A_par[$I]` (each block forced `threaded = false`), not `H.A[$I]`: this loop
# itself is the block-parallel layer, so a block that threaded internally too would nest
# its own parallelism inside it.
@generated function _hcat_block_adj!(y::Tuple, H::HCAT{N, L, P}, b, ::Val{I}, ::Val{natural}) where {N, L, P, I, natural}
    K = 0
    target = nothing
    for i in 1:N
        Pi = fieldtype(P, i)
        if Pi <: Integer
            K += 1
            i == I && (target = natural ? :(y[$K]) : :(y[H.idxs[$i]]))
        else
            n = fieldcount(Pi)
            if i == I
                parts = natural ? [:(y[$(K + j)]) for j in 1:n] :
                    [:(y[H.idxs[$i][$j]]) for j in 1:n]
                target = :(ArrayPartition($(parts...)))
            end
            K += n
        end
    end
    return :(mul!($target, H.A_par[$I]', b))
end

function _hcat_adj_mul!(
        y::Tuple, A::AdjointOperator{<:HCAT{N, L, P, C, DS, true}}, b::AbstractArray
    ) where {N, L, P, C, DS}
    H = A.A
    natural = Val(_hcat_has_natural_idxs(H))
    @budgeted_threads for i in 1:N
        _hcat_block_adj!(y, H, b, Val(i), natural)
    end
    return y
end

_hcat_has_natural_idxs(H::HCAT{N, L, P}) where {N, L, P} = H.idxs == _hcat_natural_idxs(P)

@generated function _hcat_natural_idxs(::Type{P}) where {P <: Tuple}
    K = 0
    idx_exprs = []
    for i in 1:fieldcount(P)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            K += 1
            push!(idx_exprs, K)
        else
            n = fieldcount(Pi)
            push!(idx_exprs, Expr(:tuple, (K + j for j in 1:n)...))
            K += n
        end
    end
    return Expr(:tuple, idx_exprs...)
end

# `y = α * Σᵢ Aᵢ bᵢ + β * y`: the first block applies `β`, unless `plain` (`α = 1`, `β = 0`) lets it
# write `y` directly, and every further block adds into `y`, through `H.buf` only when the block
# cannot accumulate by itself. Defined before the generated functions that call it.
function _hcat_forward_expr(N, input_expr, plain)
    ex = plain ? :(mul!(y, H.A[1], $(input_expr(1)))) : :(add_mul!(y, H.A[1], $(input_expr(1)), H.buf, α, β))
    for i in 2:N
        ex = :($ex; add_mul!(y, H.A[$i], $(input_expr(i)), H.buf, α, true))
    end
    return :($ex; return y)
end

@generated function _mul_hcat_natural!(
        y, H::HCAT{N, L, P}, b::Tuple, α = true, β = false, ::Val{plain} = Val(true)
    ) where {N, L, P, plain}
    K = 0
    function input_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            K += 1
            return :(b[$K])
        else
            n = fieldcount(Pi)
            parts = [:(b[$(K + j)]) for j in 1:n]
            K += n
            return :(ArrayPartition($(parts...)))
        end
    end

    return _hcat_forward_expr(N, input_expr, plain)
end

@generated function _mul_hcat_indexed!(
        y, H::HCAT{N, L, P}, b::Tuple, α = true, β = false, ::Val{plain} = Val(true)
    ) where {N, L, P, plain}
    function input_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            return :(b[H.idxs[$i]])
        else
            n = fieldcount(Pi)
            parts = [:(b[H.idxs[$i][$j]]) for j in 1:n]
            return :(ArrayPartition($(parts...)))
        end
    end

    return _hcat_forward_expr(N, input_expr, plain)
end

# Properties
Base.:(==)(H1::HCAT{N, L1, P1}, H2::HCAT{N, L2, P2}) where {N, L1, L2, P1, P2} = H1.A == H2.A && H1.idxs == H2.idxs

@generated function size(H::HCAT{N, L, P}) where {N, L, P}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P, i)
        if Pi <: Integer
            push!(exprs, :(size(H.A[$i], 2)))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(size(H.A[$i], 2)[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :(size(H.A[1], 1), _hcat_apply_invperm($natural_expr, H.idxs))
end

# Apply inverse permutation (from HCAT idxs) to a natural-order domain size/type tuple.
function _hcat_apply_invperm(natural::Tuple, idxs)
    p = vcat([[idx...] for idx in idxs]...)
    ip = invperm(p)
    return ntuple(j -> natural[ip[j]], Val(length(natural)))
end

function fun_name(L::HCAT)
    if length(L.A) == 2
        if L.idxs[1] == 2 || L.idxs[2] == 1
            return "[" * fun_name(L.A[2]) * "," * fun_name(L.A[1]) * "]"
        else
            return "[" * fun_name(L.A[1]) * "," * fun_name(L.A[2]) * "]"
        end
    else
        return "HCAT"
    end
end

@generated function domain_type(H::HCAT{N, L, P}) where {N, L, P}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P, i)
        if Pi <: Integer
            push!(exprs, :(domain_type(H.A[$i])))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(domain_type(H.A[$i])[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :(_hcat_apply_invperm($natural_expr, H.idxs))
end
codomain_type(L::HCAT) = codomain_type.(Ref(L.A[1]))
domain_array_type(::HCAT{N, L, P, C, DS}) where {N, L, P, C, DS} = DS
codomain_array_type(L::HCAT) = codomain_array_type.(Ref(L.A[1]))

is_linear(L::HCAT) = all(is_linear.(L.A))
is_affine(L::HCAT) = all(is_affine.(L.A))
is_AAc_diagonal(L::HCAT) = all(is_AAc_diagonal.(L.A))
is_full_row_rank(L::HCAT) = any(is_full_row_rank.(L.A))
# Columns come from independent blocks stacked side-by-side into a shared
# codomain: their sum can always cancel (e.g. HCAT(Eye,Eye) has [1,-1] in its
# kernel), so the safe answer is `false` unless proven otherwise upstream.
is_full_column_rank(L::HCAT) = false

# `‖Ax + By‖ ≤ ‖A‖‖x‖ + ‖B‖‖y‖ ≤ sqrt(‖A‖² + ‖B‖²)·sqrt(‖x‖² + ‖y‖²)`, by Cauchy–Schwarz.
opnorm_bound(L::HCAT) = sqrt(sum(abs2, opnorm_bound.(L.A)))

# HCAT(A, A, ..., A) with the *same* operator repeated (the shared-encoding-
# operator multi-component reconstruction case, e.g. HCAT(𝒜, 𝒜) for
# 𝒜*(x+y)) has AᴴA = Bᴴ * get_normal_op(A) * B, where B = HCAT(Eye,...,Eye)
# sums the blocks and Bᴴ = VCAT(Eye,...,Eye) broadcasts the result back out —
# reusing A's own fast normal operator (e.g. a Toeplitz-embedded NFFT AᴴA)
# instead of applying A and Aᴴ once per block. Only safe when every block is
# provably the same operator; a generic HCAT of distinct operators has no
# such fusion (and its off-diagonal AᴴA cross terms are, in general, not
# cheap to form), so it keeps the `false` fallback of `has_optimized_normalop`.
function has_optimized_normalop(H::HCAT)
    A1 = H.A[1]
    return all(==(A1), H.A) && has_optimized_normalop(A1)
end

function get_normal_op(H::HCAT)
    A1 = H.A[1]
    n = length(H.A)
    AtA_fast = get_normal_op(A1)
    block_eye = Eye(domain_type(A1), size(A1, 2))
    sum_op = HCAT(ntuple(_ -> block_eye, n)...)      # ArrayPartition -> sum of blocks
    broadcast_op = VCAT(ntuple(_ -> block_eye, n)...) # single block -> broadcast to ArrayPartition
    return broadcast_op * AtA_fast * sum_op
end

is_sliced(L::HCAT) = any(is_sliced.(L.A))
function get_slicing_expr(L::HCAT)
    exprs = ()
    for i in eachindex(L.A)
        exprs = (exprs..., get_slicing_expr(L.A[i]))
    end
    if length(exprs) == 1
        return exprs[1]
    else
        return exprs
    end
end
get_slicing_mask(L::HCAT) = get_slicing_mask.(L[i] for i in eachindex(L.A))
remove_slicing(L::HCAT) = HCAT(remove_slicing.(Tuple(A for A in L.A)), L.idxs, L.buf; threaded = is_block_threaded(L))

diag_AAc(L::HCAT) = (+).(diag_AAc.(L[i] for i in eachindex(L.A))...)

# utils
function permute(H::HCAT, p::AbstractVector{Int})
    unfolded = vcat([[idx...] for idx in H.idxs]...)
    invpermute!(unfolded, p)

    new_part = ()
    cnt = 0
    for z in length.(H.idxs)
        new_part = (
            new_part..., z == 1 ? unfolded[cnt + 1] : (unfolded[(cnt + 1):(z + cnt)]...,),
        )
        cnt += z
    end

    return HCAT(H.A, new_part, H.buf; threaded = is_block_threaded(H))
end

function remove_displacement(H::HCAT)
    return HCAT(remove_displacement.(H.A), H.idxs, H.buf; threaded = is_block_threaded(H))
end

function _copy_operator_impl(
        op::HCAT{N, L, P, C, DS, Th}; storage_type = nothing, threaded = nothing
    ) where {N, L, P, C, DS, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_buf = _convert_buffer(op.buf, storage_type)
    # `op.A` holds each block's own natural threading state (the constructor derives the
    # adjoint-only forced-serial copy itself), so the original `threaded` request is
    # forwarded to the children unchanged rather than forced false.
    new_ops = tuple([copy_operator(a; storage_type, threaded) for a in op.A]...)
    return HCAT(new_ops, op.idxs, new_buf; threaded = new_threaded)
end

# Whether the *adjoint block loop itself* threads, as distinct from `is_threaded`, which is
# also true when merely a child threads. Structural transforms need the former to round-trip.
# PROVENANCE: measured, and deliberately one step above DCAT/VCAT. HCAT's adjoint carries
# per-block ArrayPartition indexing that the others do not, and it shows: speedup by
# per-block size across 4/8/16 blocks is 2^15 -> 0.63/0.83/1.51, 2^16 -> 0.76/1.25/1.31
# (4 blocks still LOSES), 2^17 -> 1.27/1.19/4.25 (all win). Sharing VCAT's 2^16 would ship
# a measured 0.76x regression at 4 blocks -- this is exactly why the threshold is a
# per-operator function rather than one shared constant.
block_threading_threshold(::Type{<:HCAT}) = 2^17

is_block_threaded(::HCAT{N, L, P, C, DS, Th}) where {N, L, P, C, DS, Th} = Th

_children(L::HCAT) = L.A
is_threaded(L::HCAT{N, Ls, P, C, DS, Th}) where {N, Ls, P, C, DS, Th} =
    Th || _is_threaded_from_children(L)
supports_threading(::HCAT) = true

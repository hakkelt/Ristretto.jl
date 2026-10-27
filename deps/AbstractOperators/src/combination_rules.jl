function can_be_combined(T1::AffineAdd, T2::AffineAdd)
    return is_linear(T1.A) && can_be_combined(T1.A, T2.A)
end
function combine(T1::AffineAdd{L1, D1, S1}, T2::AffineAdd{L2, D2, S2}) where {L1, D1, S1, L2, D2, S2}
    new_d = T1.A * T2.d
    if S1 == S2
        new_d .+= T1.d
    else
        new_d .-= T1.d
    end
    return AffineAdd(combine(T1.A, T2.A), new_d, S2)
end

can_be_combined(T1, T2::AffineAdd) = is_linear(T1) && can_be_combined(T1, T2.A)
function combine(T1, T2::AffineAdd{L, D, S}) where {L, D, S}
    new_d = if T2.d isa Number
        temp = allocate_in_domain(T1)
        temp .= T2.d
        T1 * temp
    else
        T1 * T2.d
    end
    return AffineAdd(combine(T1, T2.A), new_d, S)
end

can_be_combined(L, R::Compose) = can_be_combined(L, R.A[end])
can_be_combined(L::Compose, R::Compose) = can_be_combined(L.A[1], R.A[end])
function can_be_combined(L::Scale, R::Compose)
    return can_be_combined(L.A, R.A[end]) || (
        is_linear(L.A) &&
            all(is_linear.(R.A)) &&
            any(op -> op isa Union{Scale, DiagOp, MatrixOp, LMatrixOp}, R.A)
    )
end
function can_be_combined(L::AdjointOperator{<:Scale}, R::Compose)
    return can_be_combined(L.A.A', R.A[end]) || (
        is_linear(L.A.A') &&
            all(is_linear.(R.A)) &&
            any(op -> op isa Union{Scale, DiagOp, MatrixOp, LMatrixOp}, R.A)
    )
end
function can_be_combined(L::Compose, R::Scale)
    return can_be_combined(L.A[end], R.A) || (
        is_linear(R.A) &&
            all(is_linear.(L.A)) &&
            any(op -> op isa Union{Scale, DiagOp, MatrixOp, LMatrixOp}, L.A)
    )
end
function can_be_combined(L::Compose, R::AdjointOperator{<:Scale})
    return can_be_combined(L.A[end], R.A.A') || (
        is_linear(R.A.A') &&
            all(is_linear.(L.A)) &&
            any(op -> op isa Union{Scale, DiagOp, MatrixOp, LMatrixOp}, L.A)
    )
end

function combine(L, R::Compose)
    combined = combine(L, R.A[end])
    if combined isa Compose
        ops = (R.A[1:(end - 1)]..., combined.A...)
        bufs = (R.buf..., combined.buf...)
    else
        ops = (R.A[1:(end - 1)]..., combined)
        bufs = R.buf
    end
    return Compose(ops, bufs)
end
function combine(L::Compose, R::Compose)
    combined = combine(L.A[1], R.A[end])
    if combined isa Compose
        ops = (R.A[1:(end - 1)]..., combined.A..., L.A[2:end]...)
        bufs = (R.buf..., combined.buf..., L.buf...)
    else
        ops = (R.A[1:(end - 1)]..., combined, L.A[2:end]...)
        bufs = (R.buf..., L.buf...)
    end
    return Compose(ops, bufs)
end
function combine(L::Scale{Th}, R::Compose) where {Th}
    threaded = Th == FastBroadcast.Threaded()
    if can_be_combined(L.A, R.A[end])
        return Scale(L.coeff, L.coeff_conj, combine(L.A, R); threaded) # forward optimization task to combine function
    else
        return Scale(L.coeff, L.A * R; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
    end
end
function combine(L::AdjointOperator{<:Scale{Th}}, R::Compose) where {Th}
    threaded = Th == FastBroadcast.Threaded()
    if can_be_combined(L.A.A', R.A[end])
        return Scale(L.A.coeff_conj, L.A.coeff, combine(L.A.A', R); threaded) # forward optimization task to combine function
    else
        return Scale(L.A.coeff_conj, L.A.A' * R; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
    end
end
function combine(L::Compose, R::Scale{Th}) where {Th}
    threaded = Th == FastBroadcast.Threaded()
    if can_be_combined(L.A[1], R.A)
        S = Scale(R.coeff, R.coeff_conj, combine(L.A[1], R.A); threaded) # forward optimization task to combine function
        return Compose((S, L.A[2:end]...), L.buf)
    else
        return Scale(R.coeff, L * R.A; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
    end
end
function combine(
        L::Compose, R::AdjointOperator{<:Scale{Th}}
    ) where {Th}
    threaded = Th == FastBroadcast.Threaded()
    if can_be_combined(L.A[1], R.A.A')
        S = Scale(R.A.coeff_conj, R.A.coeff, combine(L.A[1], R.A.A'); threaded) # forward optimization task to combine function
        return Compose((S, L.A[2:end]...), L.buf)
    else
        return Scale(R.A.coeff_conj, L * R.A.A'; threaded) # forward optimization task to the specialized constructor of Scale(coeff, L::Compose)
    end
end

function can_be_combined(L::DCAT, R::DCAT)
    return length(L.A) == length(R.A) &&
        L.idxD == R.idxD &&
        L.idxC == R.idxC &&
        all(can_be_combined(L.A[i], R.A[i]) for i in eachindex(L.A))
end

function combine(L::DCAT, R::DCAT)
    return DCAT([combine(L.A[i], R.A[i]) for i in 1:ndoms(L, 1)]...)
end

function can_be_combined(L1, L2::HCAT)
    return is_diagonal(L1) && all(can_be_combined(L1, A) for A in L2.A)
end
function combine(L1, L2::HCAT)
    combined = tuple([combine(L1, A) for A in L2.A]...)
    return HCAT(combined, L2.idxs, L2.buf)
end

can_be_combined(L, R::Scale) = is_linear(L) && can_be_combined(L, R.A)
can_be_combined(L, R::AdjointOperator{<:Scale}) = can_be_combined(R.A.A, L')
combine(L, R::Scale) = Scale(R.coeff, combine(L, R.A))
combine(L, R::AdjointOperator{<:Scale}) = Scale(R.A.coeff, combine(R.A.A, L'))'

function can_be_combined(L, R::Sum)
    return is_linear(L) && all(can_be_combined(L, A) for A in R.A)
end
function combine(L, R::Sum)
    ops = tuple([combine(L, A) for A in R.A]...)
    return size(L, 1) == size(L, 2) ? Sum(ops, R.bufC, R.bufD) : Sum(ops...)
end

can_be_combined(T1::DiagOp, T2::DiagOp) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::DiagOp) = true
can_be_combined(T1::DiagOp, T2::AdjointOperator{<:DiagOp}) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:DiagOp}) = true
function combine(T1::DiagOp, T2::DiagOp)
    return DiagOp(domain_type(T2), T2.dim_in, T1.d .* T2.d)
end
function combine(T1::DiagOp, T2::AdjointOperator{<:DiagOp})
    return DiagOp(domain_type(T2), size(T2, 2), T1.d .* conj.(T2.A.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::DiagOp)
    return DiagOp(domain_type(T2), size(T2, 2), conj.(T1.A.d) .* T2.d)
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:DiagOp})
    return DiagOp(domain_type(T2), size(T2, 2), conj.(T1.A.d) .* conj.(T2.A.d))
end

can_be_combined(T1, ::Eye) = true
combine(T1, ::Eye) = T1

can_be_combined(::MatrixOp, ::MatrixOp) = true
can_be_combined(::AdjointOperator{<:MatrixOp}, ::MatrixOp) = true
can_be_combined(::MatrixOp, ::AdjointOperator{<:MatrixOp}) = true
can_be_combined(::AdjointOperator{<:MatrixOp}, ::AdjointOperator{<:MatrixOp}) = true
function combine(T1::MatrixOp, T2::MatrixOp)
    return MatrixOp(domain_type(T2), size(T2, 2), T1.A * T2.A)
end
function combine(T1::MatrixOp, T2::AdjointOperator{<:MatrixOp})
    return MatrixOp(domain_type(T2), size(T2, 2), T1.A * T2.A.A')
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::MatrixOp)
    return MatrixOp(domain_type(T2), size(T2, 2), T1.A.A' * T2.A)
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:MatrixOp})
    return MatrixOp(domain_type(T2), size(T2, 2), T1.A.A' * T2.A.A')
end

can_be_combined(T1::MatrixOp, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:MatrixOp}, T2::Scale) = true
can_be_combined(T1::Scale, T2::MatrixOp) = is_linear(T1.A) || can_be_combined(T1.A, T2)
can_be_combined(T1::Scale, T2::AdjointOperator{<:MatrixOp}) = is_linear(T1.A) || can_be_combined(T1.A, T2)
can_be_combined(T1::AdjointOperator{<:Scale}, T2::MatrixOp) = true
can_be_combined(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:MatrixOp}) = true
function combine(T1::MatrixOp, T2::Scale)
    return Compose(MatrixOp(T2.coeff * T1.A), T2.A)
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::Scale)
    return Compose(MatrixOp(T2.coeff * T1.A.A'), T2.A)
end
function combine(T1::Scale, T2::MatrixOp)
    if can_be_combined(T1.A, T2)
        return Scale(T1.coeff, combine(T1.A, T2))
    else
        return Compose(T1.A, MatrixOp(T1.coeff * T2.A))
    end
end
function combine(T1::Scale, T2::AdjointOperator{<:MatrixOp})
    if can_be_combined(T1.A, T2)
        return Scale(T1.coeff, combine(T1.A, T2))
    else
        return Compose(T1.A, MatrixOp(T1.coeff * T2.A.A'))
    end
end
function combine(T1::AdjointOperator{<:Scale}, T2::MatrixOp)
    return Compose(T1.A.A', MatrixOp(T1.A.coeff_conj * T2.A))
end
function combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:MatrixOp})
    return Compose(T1.A.A', MatrixOp(T1.A.coeff_conj * T2.A.A'))
end

can_be_combined(T1::Scale, T2::DiagOp) = is_linear(T1.A) || can_be_combined(T1.A, T2)
can_be_combined(T1::AdjointOperator{<:Scale}, T2::DiagOp) = true
can_be_combined(T1::DiagOp, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:DiagOp}, T2::Scale) = true
can_be_combined(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp}) = true
function combine(T1::Scale, T2::DiagOp)
    if can_be_combined(T1.A, T2)
        return Scale(T1.coeff, combine(T1.A, T2))
    else
        scaled_diagop = T1.coeff * T2
        return T1.A * scaled_diagop
    end
end
function combine(T1::AdjointOperator{<:Scale}, T2::DiagOp)
    scaled_diagop = DiagOp(domain_type(T2), size(T2, 2), T1.A.coeff_conj * T2.d)
    return if can_be_combined(T1.A.A', scaled_diagop)
        combine(T1.A.A', scaled_diagop)
    else
        T1.A.A' * scaled_diagop
    end
end
function combine(T1::DiagOp, T2::Scale)
    scaled_diagop = DiagOp(domain_type(T1), size(T1, 2), T1.d .* T2.coeff)
    return if can_be_combined(scaled_diagop, T2.A)
        combine(scaled_diagop, T2.A)
    else
        scaled_diagop * T2.A
    end
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::Scale)
    scaled_diagop = DiagOp(domain_type(T1), size(T1, 2), conj.(T1.A.d) .* T2.coeff)
    return if can_be_combined(scaled_diagop, T2.A)
        combine(scaled_diagop, T2.A)
    else
        scaled_diagop * T2.A
    end
end
function combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp})
    scaled_diagop = DiagOp(domain_type(T2), size(T2, 2), T1.A.coeff_conj .* conj.(T2.A.d))
    return if can_be_combined(T1.A.A', scaled_diagop)
        combine(T1.A.A', scaled_diagop)
    else
        T1.A.A' * scaled_diagop
    end
end

"""
	_has_matrix_diagonal(L)

Whether `L` is a `DiagOp` whose diagonal is not a vector, i.e. one that scales a multi-dimensional
array elementwise.

Such a `DiagOp` cannot be folded into a neighbouring `MatrixOp`. A `MatrixOp` with a
multi-dimensional domain applies its matrix to every column of the input independently, so
`x -> M * (d .* x)` weights each column by a *different* diagonal; that is block diagonal with
unequal blocks, and no single matrix represents it. Folding it anyway silently produced a
different operator — `M * d` as an ordinary matrix product — which agreed with the composition on
nothing.
"""
# Only a diagonal can be matrix-valued; for anything else the question does not arise.
_has_matrix_diagonal(::AbstractOperator) = false
_has_matrix_diagonal(L::DiagOp) = !(L.d isa AbstractVector)
_has_matrix_diagonal(L::AdjointOperator{<:DiagOp}) = _has_matrix_diagonal(L.A)

function _matrix_diag_combinable(T1, T2)
    return codomain_type(T1) == domain_type(T2) &&
        !_has_matrix_diagonal(T1) &&
        !_has_matrix_diagonal(T2)
end

# Either order, either factor adjointed: the pair folds under the same condition every time.
for (S1, S2) in Iterators.product(
        (:DiagOp, :(AdjointOperator{<:DiagOp})), (:MatrixOp, :(AdjointOperator{<:MatrixOp}))
    )
    @eval can_be_combined(T1::$S1, T2::$S2) = _matrix_diag_combinable(T1, T2)
    @eval can_be_combined(T1::$S2, T2::$S1) = _matrix_diag_combinable(T1, T2)
end
# Only a vector diagonal turns into a matrix factor. The `(AbstractMatrix, AbstractMatrix)`
# method that used to sit here existed solely to serve a matrix-valued diagonal, which
# `_has_matrix_diagonal` now keeps out of `combine` altogether.
combine_matrix(L::AbstractMatrix, R::AbstractVector) = L * Diagonal(R)
combine_matrix(L::AbstractVector, R::AbstractMatrix) = Diagonal(L) * R
function combine(T1::DiagOp, T2::MatrixOp)
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.d, T2.A))
end
function combine(T1::MatrixOp, T2::DiagOp)
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A, T2.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::MatrixOp)
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(conj(T1.A.d), T2.A))
end
function combine(T1::DiagOp, T2::AdjointOperator{<:MatrixOp})
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.d, T2.A.A'))
end
function combine(T1::MatrixOp, T2::AdjointOperator{<:DiagOp})
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A, conj(T2.A.d)))
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::DiagOp)
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A.A', T2.d))
end
function combine(T1::AdjointOperator{<:DiagOp}, T2::AdjointOperator{<:MatrixOp})
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(conj(T1.A.d), T2.A.A'))
end
function combine(T1::AdjointOperator{<:MatrixOp}, T2::AdjointOperator{<:DiagOp})
    return MatrixOp(domain_type(T2), size(T2, 2), combine_matrix(T1.A.A', conj(T2.A.d)))
end

# Batch transparency
#
# A `BatchOp` applies its operators slice by slice over its batch dimensions and mixes nothing
# between slices, so a composition that crosses a batch boundary could always have been written
# with the boundary one factor further out. Two rules follow.
#
# Two batches over the same batching combine when their stored operators do, slice for slice. A
# `SimpleBatchOp` stores one operator for every slice and a `SpreadingBatchOp` one per spreading
# index, which is the same statement with a differently shaped operator array — represented here
# as a zero-dimensional array in the simple case, so that broadcasting pairs the two up.
#
# An operator that is separable over the same dimensions can be pushed inside the batch and
# combined there. `_slice_operator` is what recognises such an operator and returns its per-slice
# factor.

"""
	_slice_operator(L, batch_dim_mask) -> AbstractOperator or nothing

The per-slice factor of `L` when `L` acts separately and identically on every slice taken along
the dimensions `batch_dim_mask` marks, or `nothing` when it does not (the default).

This is what lets a plain operator be pushed inside a `BatchOp` so that the two can combine.
`nothing` is always a legal answer and merely leaves the composition unfolded, so a method is
worth adding only where separability is structural and cheap to act on: it is called while
combination rules are being *tried*, so a method that builds an expensive object — an FFT plan,
say — pays for that on every speculative check. A method that claims separability wrongly yields
a silently wrong operator, so err towards `nothing`.
"""
_slice_operator(::AbstractOperator, ::NTuple{N, Bool}) where {N} = nothing

# A scaling is separable exactly when what it scales is.
function _slice_operator(L::Scale, mask::NTuple{N, Bool}) where {N}
    inner = _slice_operator(L.A, mask)
    # `_rethread_scale` carries the threading flag over, as every other rule that rebuilds a
    # `Scale` does. The constructor's default is `threaded = true`, which is the wrong answer
    # inside a batch loop that is itself the parallel layer.
    return inner === nothing ? nothing : _rethread_scale(L, L.coeff, L.coeff_conj, inner)
end

# Taking the adjoint acts within a slice, so it preserves separability.
function _slice_operator(L::AdjointOperator, mask::NTuple{N, Bool}) where {N}
    inner = _slice_operator(L.A, mask)
    return inner === nothing ? nothing : inner'
end

# A sum is separable when every term is, over the same slices.
function _slice_operator(L::Sum, mask::NTuple{N, Bool}) where {N}
    slices = map(op -> _slice_operator(op, mask), L.A)
    any(s -> s === nothing, slices) && return nothing
    return Sum(slices...)
end

# A diagonal is separable when its diagonal repeats along the batch dimensions: every slice is
# then weighted by the same face of it. A scalar diagonal repeats trivially.
function _slice_operator(L::DiagOp, mask::NTuple{N, Bool}) where {N}
    length(L.dim_in) == N || return nothing
    slice_dim_in = Tuple(L.dim_in[i] for i in 1:N if !mask[i])
    if !(L.d isa AbstractArray)
        return DiagOp(
            domain_type(L), slice_dim_in, L.d;
            threaded = is_threaded(L), array_type = domain_array_type(L),
        )
    end
    face = view(L.d, ntuple(i -> mask[i] ? 1 : Colon(), N)...)
    for I in CartesianIndices(ntuple(i -> mask[i] ? axes(L.d, i) : Base.OneTo(1), N))
        other = view(L.d, ntuple(i -> mask[i] ? I[i] : Colon(), N)...)
        other == face || return nothing
    end
    return DiagOp(
        domain_type(L), slice_dim_in, copy(face);
        threaded = is_threaded(L), array_type = domain_array_type(L),
    )
end

# What a batch stores, as an array indexed by its spreading dimensions. A `SimpleBatchOp` applies
# the same operator to every slice, which is the zero-dimensional case: it broadcasts against any
# spreading grid, so one set of rules covers both families.
_batch_operators(L::SimpleBatchOp) = fill(_wrapped_operator(L))
_batch_operators(L::SpreadingBatchOp) = _spreading_operators(L)

_spreading_dims_of(::SimpleBatchOp) = ()
_spreading_dims_of(L::SpreadingBatchOp) = get_spreading_dims(typeof(L))

_batch_threading_strategy(::SimpleBatchOp) = ThreadingStrategy.AUTO
_batch_threading_strategy(L::SpreadingBatchOp) = _threading_strategy_of(L)

# The batchings line up when the shared face — `L`'s domain against `R`'s codomain — agrees in
# size, in which dimensions are batch dimensions, and in how many slices there are.
function _same_batching(L::BatchOp, R::BatchOp)
    return _batch_size(L) == _batch_size(R) &&
        L.domain_size == R.codomain_size &&
        get_domain_batch_dim_mask(typeof(L)) == get_codomain_batch_dim_mask(typeof(R))
end

# Which dimensions select an operator in the combined batch. A side that spreads over nothing
# contributes nothing, and two sides that spread over different dimensions have no common grid,
# which `nothing` reports.
function _combined_spreading_dims(L::BatchOp, R::BatchOp)
    sL, sR = _spreading_dims_of(L), _spreading_dims_of(R)
    isempty(sL) && return sR
    isempty(sR) && return sL
    return sL == sR ? sL : nothing
end

# A batch that spreads keeps its strategy; two that disagree fall back to letting the constructor
# choose, which is what `AUTO` means.
function _combined_threading_strategy(L::BatchOp, R::BatchOp)
    sL, sR = _batch_threading_strategy(L), _batch_threading_strategy(R)
    sL === ThreadingStrategy.AUTO && return sR
    sR === ThreadingStrategy.AUTO && return sL
    return sL === sR ? sL : ThreadingStrategy.AUTO
end

# Rebuild a batch around `operators`, zero-dimensional for one operator per slice. `create_BatchOp`
# derives both array shapes from the operators and the batch size and asserts that they fit the
# masks, so an operator of the wrong rank fails here rather than producing a mis-shaped batch.
function _rebatch(
        operators::AbstractArray, batch_size, domain_mask, codomain_mask, spreading_dims,
        threaded, threading_strategy,
    )
    if ndims(operators) == 0
        return create_BatchOp(
            operators[], batch_size, domain_mask => codomain_mask; threaded
        )
    end
    domain_size, domain_mask, codomain_size, codomain_mask = calculate_shapes(
        operators[1], batch_size, domain_mask => codomain_mask
    )
    return create_BatchOp(
        operators, domain_size, domain_mask, codomain_size, codomain_mask, spreading_dims;
        threaded, threading_strategy,
    )
end

# Pair up two grids of operators. `broadcast` collapses to a plain value when every argument is
# zero-dimensional, which would lose the "one operator for every slice" shape that `_rebatch`
# dispatches on, so that case is spelled out.
_pair_operators(f, L::AbstractArray{<:Any, 0}, R::AbstractArray{<:Any, 0}) = fill(f(L[], R[]))
_pair_operators(f, L, R) = f.(L, R)

function can_be_combined(L::BatchOp, R::BatchOp)
    _same_batching(L, R) || return false
    _combined_spreading_dims(L, R) === nothing && return false
    return all(_pair_operators(can_be_combined, _batch_operators(L), _batch_operators(R)))
end
function combine(L::BatchOp, R::BatchOp)
    return _rebatch(
        _pair_operators(combine, _batch_operators(L), _batch_operators(R)),
        _batch_size(R),
        get_domain_batch_dim_mask(typeof(R)),
        get_codomain_batch_dim_mask(typeof(L)),
        _combined_spreading_dims(L, R),
        is_threaded(L) || is_threaded(R),
        _combined_threading_strategy(L, R),
    )
end

# Only a shape-preserving neighbour is pushed into a batch: the batch carries one dimension mask
# per side, and a slice factor that changed the slice's rank would need a different one.
function _absorbed_slice(outer, mask, face_size)
    size(outer, 1) == face_size && size(outer, 2) == face_size || return nothing
    return _slice_operator(outer, mask)
end

function can_be_combined(L::AbstractOperator, R::BatchOp)
    # An identity or a null operator is dropped by the generic rule, whichever side it is on, and
    # that answer stays right for a batch.
    invoke(can_be_combined, Tuple{Any, Any}, L, R) && return true
    slice = _absorbed_slice(L, get_codomain_batch_dim_mask(typeof(R)), R.codomain_size)
    slice === nothing && return false
    return all(op -> can_be_combined(slice, op), _batch_operators(R))
end
function combine(L::AbstractOperator, R::BatchOp)
    mask = get_codomain_batch_dim_mask(typeof(R))
    slice = _absorbed_slice(L, mask, R.codomain_size)
    slice === nothing && return invoke(combine, Tuple{Any, Any}, L, R)
    return _rebatch(
        map(op -> combine(slice, op), _batch_operators(R)),
        _batch_size(R),
        get_domain_batch_dim_mask(typeof(R)),
        mask,
        _spreading_dims_of(R),
        is_threaded(R),
        _batch_threading_strategy(R),
    )
end

function can_be_combined(L::BatchOp, R::AbstractOperator)
    invoke(can_be_combined, Tuple{Any, Any}, L, R) && return true
    slice = _absorbed_slice(R, get_domain_batch_dim_mask(typeof(L)), L.domain_size)
    slice === nothing && return false
    return all(op -> can_be_combined(op, slice), _batch_operators(L))
end
function combine(L::BatchOp, R::AbstractOperator)
    mask = get_domain_batch_dim_mask(typeof(L))
    slice = _absorbed_slice(R, mask, L.domain_size)
    slice === nothing && return invoke(combine, Tuple{Any, Any}, L, R)
    return _rebatch(
        map(op -> combine(op, slice), _batch_operators(L)),
        _batch_size(L),
        mask,
        get_codomain_batch_dim_mask(typeof(L)),
        _spreading_dims_of(L),
        is_threaded(L),
        _batch_threading_strategy(L),
    )
end

# Disambiguation against the rules above that dispatch on their *second* argument alone. In each
# case that other rule is the one that applies: a batch is not special on the side where it plays
# the role of a plain factor.
for OtherT in (:Compose, :Scale, :Sum, :HCAT, :Eye)
    @eval begin
        function can_be_combined(L::BatchOp, R::$OtherT)
            return invoke(can_be_combined, Tuple{Any, $OtherT}, L, R)
        end
        combine(L::BatchOp, R::$OtherT) = invoke(combine, Tuple{Any, $OtherT}, L, R)
    end
end
# `AffineAdd` is spelled out rather than generated: its `combine` carries static parameters, and a
# disambiguating method only counts as one when its signature is written the same way.
can_be_combined(L::BatchOp, R::AffineAdd) = invoke(can_be_combined, Tuple{Any, AffineAdd}, L, R)
function combine(L::BatchOp, R::AffineAdd{L2, D2, S2}) where {L2, D2, S2}
    return invoke(combine, Tuple{Any, AffineAdd{L2, D2, S2}}, L, R)
end
can_be_combined(L::BatchOp, R::AdjointOperator{<:Scale}) = can_be_combined(R.A.A, L')
combine(L::BatchOp, R::AdjointOperator{<:Scale}) = Scale(R.A.coeff, combine(R.A.A, L'))'

# Forward differences along one direction compose into one of the summed order: `L * R` with
# `R` applied first is `HigherOrderDiff` of order `K_L + K_R` on `R`'s domain, and the adjoints
# `L' * R'` are `(R * L)'`. Both must share direction, element type and storage. The result is
# threaded when either operator was, subject to the policy at its own size.
const _ForwardDiff = Union{FiniteDiff, HigherOrderDiff}
_diff_direction(::FiniteDiff{N, D}) where {N, D} = D
_diff_direction(::HigherOrderDiff{N, D}) where {N, D} = D
_diff_order(::FiniteDiff) = 1
_diff_order(::HigherOrderDiff{N, D, K}) where {N, D, K} = K

function can_be_combined(L::_ForwardDiff, R::_ForwardDiff)
    return _diff_direction(L) == _diff_direction(R) &&
        domain_type(L) == domain_type(R) &&
        domain_array_type(L) == domain_array_type(R) &&
        size(R, 1) == size(L, 2)
end
function can_be_combined(L::AdjointOperator{<:_ForwardDiff}, R::AdjointOperator{<:_ForwardDiff})
    return can_be_combined(R.A, L.A)
end
function combine(L::_ForwardDiff, R::_ForwardDiff)
    return HigherOrderDiff(
        domain_type(R), size(R, 2), Val(_diff_direction(R)), Val(_diff_order(L) + _diff_order(R));
        array_type = domain_array_type(R), threaded = is_threaded(L) || is_threaded(R)
    )
end
combine(L::AdjointOperator{<:_ForwardDiff}, R::AdjointOperator{<:_ForwardDiff}) = combine(R.A, L.A)'

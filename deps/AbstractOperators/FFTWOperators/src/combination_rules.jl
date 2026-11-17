# DCT and IDCT
can_be_combined(T1::IDCT, T2::DCT) = true
can_be_combined(T1::DCT, T2::IDCT) = true
can_be_combined(T1::DCT, T2::AdjointOperator{<:DCT}) = true
can_be_combined(T1::IDCT, T2::AdjointOperator{<:IDCT}) = true
can_be_combined(T1::AdjointOperator{<:DCT}, T2::IDCT) = true
can_be_combined(T1::AdjointOperator{<:IDCT}, T2::DCT) = true
can_be_combined(T1::AdjointOperator{<:DCT}, T2::AdjointOperator{<:IDCT}) = true
can_be_combined(T1::AdjointOperator{<:IDCT}, T2::AdjointOperator{<:DCT}) = true
combine(::CosineTransform, T2::CosineTransform) = Eye(allocate_in_domain(T2))
function combine(::CosineTransform, T2::AdjointOperator{<:CosineTransform})
    return Eye(allocate_in_domain(T2))
end
function combine(::AdjointOperator{<:CosineTransform}, T2::CosineTransform)
    return Eye(allocate_in_domain(T2))
end
function combine(
        ::AdjointOperator{<:CosineTransform}, T2::AdjointOperator{<:CosineTransform}
    )
    return Eye(allocate_in_domain(T2))
end

# DFT
can_be_combined(T1::DFT{N, C, D, Dir}, T2::AdjointOperator{<:DFT{N, C, D, Dir}}) where {N, C, D, Dir} = true
can_be_combined(T1::AdjointOperator{<:DFT{N, C, D, Dir}}, T2::DFT{N, C, D, Dir}) where {N, C, D, Dir} = true
function combine(T1::DFT{N, C, D, Dir}, T2::AdjointOperator{<:DFT}) where {N, C, D, Dir}
    scaling = _dft_scaling(T1.dim_in, Dir, FORWARD)
    scaling /= _dft_scaling(T1.dim_in, Dir, T1.normalization)
    scaling /= _dft_scaling(T1.dim_in, Dir, T2.A.normalization)
    return scaling * Eye(domain_type(T2), T1.dim_in; array_type = domain_array_type(T2))
end
function combine(T1::AdjointOperator{<:DFT}, T2::DFT{N, C, D, Dir}) where {N, C, D, Dir}
    scaling = _dft_scaling(T2.dim_in, Dir, FORWARD)
    scaling /= _dft_scaling(T2.dim_in, Dir, T1.A.normalization)
    scaling /= _dft_scaling(T2.dim_in, Dir, T2.normalization)
    return scaling * Eye(domain_type(T2), T2.dim_in; array_type = domain_array_type(T2))
end

# FFTShift/IFFTShift with DFT: the shift becomes a phase ramp on the other side of the DFT
# (`SignAlternation` when every shifted length is even, a `DiagOp` otherwise; see Shift.jl).
# `F ∘ S`: a shift of the DFT's input.
function _can_absorb_input_shift(F, S)
    return _ramp_absorbs(F, _shift_dirs(S), :domain)
end
function _absorb_input_shift(F, S)
    dirs = _shift_dirs(S)
    if _all_even(size(F, 2), dirs)
        return SignAlternation(codomain_type(F), size(F, 1), dirs; threaded = is_threaded(F), array_type = codomain_array_type(F)) * F
    end
    return _phase_ramp_op(allocate_in_codomain(F), dirs, n -> _shift_amount(S, n), _dft_sign(F); threaded = is_threaded(F)) * F
end
# `S ∘ F`: a shift of the DFT's output.
function _can_absorb_output_shift(S, F)
    dirs = _shift_dirs(S)
    return all(in(_dft_dirs(F)), dirs) && (_all_even(size(F, 1), dirs) || domain_type(F) <: Complex)
end
function _absorb_output_shift(S, F)
    dirs = _shift_dirs(S)
    if _all_even(size(F, 1), dirs)
        return F * SignAlternation(domain_type(F), size(F, 2), dirs; threaded = is_threaded(F), array_type = domain_array_type(F))
    end
    return F * _phase_ramp_op(allocate_in_domain(F), dirs, n -> _shift_amount(S, n), -_dft_sign(F); threaded = is_threaded(F))
end
# Every shifted length even and every shifted dimension transformed: the rules with a
# `SignAlternation` between the shift and the DFT below need both.
function _even_shift_next_to_dft(F, S)
    dirs = _shift_dirs(S)
    return all(in(_dft_dirs(F)), dirs) && _all_even(size(F, 1), dirs)
end

# Moving a shift by N/2 past a sign alternation along the same dimension of even length N
# multiplies by (-1)^(N/2): the rules below that swap the two carry this sign.
function _sign_past_alternation(S, A::SignAlternation, op)
    m = sum((A.dim_in[d] ÷ 2 for d in _shift_dirs(S) if d in A.dirs); init = 0)
    return iseven(m) ? op : -one(real(codomain_type(op))) * op
end

for TF in (DFT, AdjointOperator{<:DFT}), TS in (ShiftOp, AdjointOperator{<:ShiftOp})
    @eval begin
        can_be_combined(T1::$TF, T2::$TS) = _can_absorb_input_shift(T1, T2)
        can_be_combined(T1::$TS, T2::$TF) = _can_absorb_output_shift(T1, T2)
        combine(T1::$TF, T2::$TS) = _absorb_input_shift(T1, T2)
        combine(T1::$TS, T2::$TF) = _absorb_output_shift(T1, T2)
    end
end

# FFTShift/IFFTShift with DFT and SignAlternation
function can_be_combined(T1::ShiftOp, ::SignAlternation, T3::DFT)
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::DFT, ::SignAlternation, T3::ShiftOp)
    return _even_shift_next_to_dft(T1, T3)
end
function can_be_combined(T1::AdjointOperator{<:ShiftOp}, ::SignAlternation, T3::DFT)
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::DFT, ::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return _even_shift_next_to_dft(T1, T3)
end
function can_be_combined(T1::ShiftOp, ::SignAlternation, T3::AdjointOperator{<:DFT})
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::AdjointOperator{<:DFT}, ::SignAlternation, T3::ShiftOp)
    return _even_shift_next_to_dft(T1, T3)
end
function can_be_combined(T1::AdjointOperator{<:ShiftOp}, ::SignAlternation, T3::AdjointOperator{<:DFT})
    return _even_shift_next_to_dft(T3, T1)
end
function can_be_combined(T1::AdjointOperator{<:DFT}, ::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return _even_shift_next_to_dft(T1, T3)
end
function combine(T1::ShiftOp, T2::SignAlternation, T3::DFT)
    return _sign_past_alternation(T1, T2, T2 * combine(T1, T3))
end
function combine(T1::DFT, T2::SignAlternation, T3::ShiftOp)
    return _sign_past_alternation(T3, T2, combine(T1, T3) * T2)
end
function combine(T1::AdjointOperator{<:ShiftOp}, T2::SignAlternation, T3::DFT)
    return _sign_past_alternation(T1, T2, T2 * combine(T1, T3))
end
function combine(T1::DFT, T2::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return _sign_past_alternation(T3, T2, combine(T1, T3) * T2)
end
function combine(T1::ShiftOp, T2::SignAlternation, T3::AdjointOperator{<:DFT})
    return _sign_past_alternation(T1, T2, T2 * combine(T1, T3))
end
function combine(T1::AdjointOperator{<:DFT}, T2::SignAlternation, T3::ShiftOp)
    return _sign_past_alternation(T3, T2, combine(T1, T3) * T2)
end
function combine(T1::AdjointOperator{<:ShiftOp}, T2::SignAlternation, T3::AdjointOperator{<:DFT})
    return _sign_past_alternation(T1, T2, T2 * combine(T1, T3))
end
function combine(T1::AdjointOperator{<:DFT}, T2::SignAlternation, T3::AdjointOperator{<:ShiftOp})
    return _sign_past_alternation(T3, T2, combine(T1, T3) * T2)
end

# FFTShift/IFFTShift with FFTShift/IFFTShift
have_shifted_dims_even_length(T::ShiftOp) = all(iseven, size(T, 2)[collect(T.dirs)])
are_shifted_dims_disjoint(T1::ShiftOp, T2::ShiftOp) =
    all(d -> !(d in T1.dirs && d in T2.dirs), 1:ndims(T1, 2))
does_fully_cover(T1::ShiftOp, T2::ShiftOp) = all(d -> (d in T1.dirs && d in T2.dirs), T1.dirs) # T1 fully covers T2
function can_be_combined(T1::FFTShift, T2::FFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || are_shifted_dims_disjoint(T1, T2)
end
function can_be_combined(T1::IFFTShift, T2::IFFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || are_shifted_dims_disjoint(T1, T2)
end
function can_be_combined(T1::FFTShift, T2::IFFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || does_fully_cover(T1, T2) || does_fully_cover(T2, T1)
end
function can_be_combined(T1::IFFTShift, T2::FFTShift)
    return (have_shifted_dims_even_length(T1) && have_shifted_dims_even_length(T2)) || does_fully_cover(T1, T2) || does_fully_cover(T2, T1)
end
function combine(T1::FFTShift, T2::FFTShift)
    new_dirs = Tuple(d for d in 1:ndims(T1, 2) if (d in T1.dirs) || (d in T2.dirs))
    return FFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
end
function combine(T1::IFFTShift, T2::IFFTShift)
    new_dirs = Tuple(d for d in 1:ndims(T1, 2) if (d in T1.dirs) || (d in T2.dirs))
    return IFFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
end
function combine(T1::FFTShift, T2::IFFTShift)
    if does_fully_cover(T1, T2)
        new_dirs = Tuple(d for d in T1.dirs if !(d in T2.dirs))
        if isempty(new_dirs)
            return Eye(domain_type(T1), size(T1, 2); array_type = domain_array_type(T1))
        else
            return FFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
        end
    else
        new_dirs = Tuple(d for d in T2.dirs if !(d in T1.dirs))
        return IFFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
    end
end
function combine(T1::IFFTShift, T2::FFTShift)
    if does_fully_cover(T1, T2)
        new_dirs = Tuple(d for d in T1.dirs if !(d in T2.dirs))
        if isempty(new_dirs)
            return Eye(domain_type(T1), size(T1, 2); array_type = domain_array_type(T1))
        else
            return IFFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
        end
    else
        new_dirs = Tuple(d for d in T2.dirs if !(d in T1.dirs))
        return FFTShift(domain_type(T1), size(T1, 2), new_dirs; array_type = domain_array_type(T1))
    end
end

# SignAlternation with SignAlternation
can_be_combined(T1::SignAlternation, T2::SignAlternation) = true
function combine(T1::SignAlternation, T2::SignAlternation)
    new_dirs = Tuple(d for d in 1:ndims(T1, 1) if (d in T1.dirs) != (d in T2.dirs))
    if isempty(new_dirs)
        return Eye(domain_type(T1), size(T1, 2); array_type = domain_array_type(T1))
    else
        return SignAlternation(
            domain_type(T1), size(T1, 2), new_dirs;
            threaded = is_threaded(T1) || is_threaded(T2), array_type = domain_array_type(T1),
        )
    end
end

# SignAlternation with DiagOp
#
# The signs are folded into the diagonal, so only the `DiagOp`'s pass is left, and it keeps the
# threading it was built with: a diagonal built serial (typically because it runs inside a
# threaded batch loop) must not come back threaded because a sign alternation was absorbed.
can_be_combined(::SignAlternation, T2::DiagOp) = diag(T2) isa AbstractArray
can_be_combined(T1::DiagOp, ::SignAlternation) = diag(T1) isa AbstractArray
function combine(T1::SignAlternation, T2::DiagOp)
    return DiagOp(domain_type(T2), size(T2, 2), T1 * diag(T2); threaded = is_threaded(T2))
end
function combine(T1::DiagOp, T2::SignAlternation)
    return DiagOp(domain_type(T1), size(T1, 2), T2 * diag(T1); threaded = is_threaded(T1))
end

# Shift operators inside a batch
#
# What a `SignAlternation` or a shift does at an index depends only on the coordinates along its
# `dirs`. Along every other dimension it acts identically, so as long as none of `dirs` is a batch
# dimension the operator is one and the same pattern applied to each slice, and that pattern is
# the same operator over the slice's own dimensions. Neither builds a plan, so recognising this
# stays cheap enough to do while combination rules are only being tried.

# The slice's dimensions, and where each of `dirs` ends up among them, or `nothing` when `dirs`
# reaches a batch dimension and the operator therefore differs from slice to slice.
function _sliced_dims_and_dirs(dim_in::NTuple{N, Int}, dirs, batch_dim_mask::NTuple{N, Bool}) where {N}
    any(d -> batch_dim_mask[d], dirs) && return nothing
    slice_dim_in = Tuple(dim_in[d] for d in 1:N if !batch_dim_mask[d])
    slice_position = cumsum(.!batch_dim_mask)
    return slice_dim_in, Tuple(slice_position[d] for d in dirs)
end

function _slice_operator(
        L::SignAlternation{T, N, M, Th, S}, batch_dim_mask::NTuple{N, Bool}
    ) where {T, N, M, Th, S}
    sliced = _sliced_dims_and_dirs(L.dim_in, L.dirs, batch_dim_mask)
    sliced === nothing && return nothing
    slice_dim_in, slice_dirs = sliced
    return SignAlternation(
        T, slice_dim_in, slice_dirs; threaded = is_threaded(L), array_type = S
    )
end

# The two shifts differ only in which direction they rotate, which the slice inherits unchanged.
for Op in (:FFTShift, :IFFTShift)
    @eval function _slice_operator(
            L::$Op{T, N, M, S}, batch_dim_mask::NTuple{N, Bool}
        ) where {T, N, M, S}
        sliced = _sliced_dims_and_dirs(L.dim_in, L.dirs, batch_dim_mask)
        sliced === nothing && return nothing
        slice_dim_in, slice_dirs = sliced
        return $Op(T, slice_dim_in, slice_dirs; array_type = S)
    end
end

# SignAlternation ∘ (any square diagonal) ∘ SignAlternation
#
# A `SignAlternation` is a real ±1 diagonal and is its own inverse, and diagonals commute,
# so `± M ± = M M±± = M` exactly, for any square diagonal `M` carrying the same `dirs`.
#
# This is what un-fuses the encoding operator's normal operator: with
# `𝒜 = 𝒫 ∘ ± ∘ ℱ ∘ 𝒮`, `𝒜ᴴ𝒜` folds `𝒫ᴴ𝒫` into a single diagonal mask and leaves
# `(…, ℱ, ±, 𝒫ᴴ𝒫, ±, ℱᴴ, …)` — a `±` pair the pairwise cancellation cannot see because
# the mask sits between them. Every normal-operator application would otherwise pay two
# full sign passes it does not owe.
#
# `L.dirs == R.dirs` is the guard that matters: two alternations over different dimension
# sets do not cancel (their product is the alternation over the symmetric difference).
function can_be_combined(L::SignAlternation, M::AbstractOperator, R::SignAlternation)
    return L.dirs == R.dirs && L.dim_in == R.dim_in &&
        is_linear(M) && is_diagonal(M) && size(M, 1) == size(M, 2)
end
combine(::SignAlternation, M::AbstractOperator, ::SignAlternation) = M

# `c2-cancel-sign-alternation-pair` also pushed a `SignAlternation` into a `SimpleBatchOp`
# directly, through a `_push_sign_into_batch` helper. `_slice_operator` above subsumes it: the
# generic batch rules in `AbstractOperators/src/combination_rules.jl` apply to both batch
# families and to every operator that declares a slice factor, not to `SignAlternation` and
# `SimpleBatchOp` alone. The narrower methods would shadow the general ones, so they are dropped
# here.

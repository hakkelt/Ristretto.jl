export FFTShift,
    IFFTShift, SignAlternation, fftshift_op, ifftshift_op, alternate_sign, alternate_sign!

abstract type ShiftOp <: LinearOperator end

"""
    FFTShift([T::Type=Float64,] dim_in::Tuple, [dirs])
    FFTShift(dim_in...)

Creates a `LinearOperator` that permutes the array like `FFTW.fftshift` over the given `dirs`.
`dirs` must contain at least one dimension; each must be within `1:length(dim_in)`.

```jldoctest
julia> using FFTWOperators

julia> A = FFTShift((4,)); x = collect(Float64, 1:4);

julia> A * x
4-element Vector{Float64}:
 3.0
 4.0
 1.0
 2.0

julia> B = IFFTShift((4,)); B * (A * x) == x
true

julia> A2 = FFTShift((2,2), (1,2)); A2 * collect(reshape(1.0:4.0, 2, 2))
2×2 Matrix{Float64}:
 4.0  2.0
 3.0  1.0
```
"""
struct FFTShift{T, N, M, S <: AbstractArray{T}} <: ShiftOp
    dim_in::NTuple{N, Int}
    dirs::NTuple{M, Int}
end

"""
    IFFTShift([T::Type=Float64,] dim_in::Tuple, [dirs])
    IFFTShift(dim_in...)

Creates a `LinearOperator` that permutes the array like `FFTW.ifftshift` over `dirs`.
`dirs` must contain at least one dimension; each must be within `1:length(dim_in)`.

```jldoctest
julia> using FFTWOperators

julia> A = FFTShift((4,)); B = IFFTShift((4,)); x = collect(1.0:4.0);

julia> B * (A * x) == x
true

julia> B2 = IFFTShift((2,2), (1,2)); B2 * [4.0 3.0; 2.0 1.0] == [1.0 2.0; 3.0 4.0]
true
```
"""
struct IFFTShift{T, N, M, S <: AbstractArray{T}} <: ShiftOp
    dim_in::NTuple{N, Int}
    dirs::NTuple{M, Int}
end

"""
    SignAlternation([T::Type=Float64,] dim_in::Tuple, dirs)

Creates a `LinearOperator` that multiplies entries by -1 on indices where the parity sum across `dirs` is odd.
`dirs` must contain at least one dimension; each must be within `1:length(dim_in)`.

# Why is it useful?
Due to the properties of the discrete Fourier transform, when there is an even number of points along a dimension,
alternating the sign of the entries along the domain of the Fourier transform (i.e., multiplying by -1 at every
other index) is equivalent to a half-sample shift in the frequency domain.
One can see this by applying the shift theorem of the Fourier transform:  
if ``\\mathcal{F}({x_n})_k = X_k``  
then ``\\mathcal{F}({x_n \\cdot e^{\\frac{i 2\\pi}{N}n m}})_k = \\mathcal{F}({x_n \\cdot (-1)^n })_k = X_{k - N/2}``  
where `N` is the number of samples along that dimension.

# Examples
```jldoctest
julia> using FFTWOperators

julia> S = SignAlternation((4,), 1); S * collect(1.0:4.0)
4-element Vector{Float64}:
  1.0
 -2.0
  3.0
 -4.0

julia> S2 = SignAlternation((2,2), (1,2)); S2 * ones(2,2)
2×2 Matrix{Float64}:
  1.0  -1.0
 -1.0   1.0
```
"""
struct SignAlternation{T, N, M, Th, S <: AbstractArray{T}} <: LinearOperator
    dim_in::NTuple{N, Int}
    dirs::NTuple{M, Int}
end

function _normalize_dirs(::NTuple{N, Int}, dirs) where {N}
    d = Tuple(dirs)
    isempty(d) && throw(ArgumentError("dirs must contain at least one dimension"))
    all(1 <= x <= N for x in d) || throw(ArgumentError("dirs must be in 1:$N"))
    return d
end

function _make_shift(::Type{Op}, ::Type{T}, dim_in::NTuple{N, Int}, dirs, array_type::Type) where {Op, T <: Number, N}
    d = _normalize_dirs(dim_in, dirs)
    S = _normalize_array_type(array_type, T)
    return Op{T, N, length(d), S}(dim_in, d)
end

function FFTShift(
        ::Type{T}, dim_in::NTuple{N, Int}, dirs;
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return _make_shift(FFTShift, T, dim_in, dirs, array_type)
end
function FFTShift(
        ::Type{T}, dim_in::NTuple{N, Int};
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return FFTShift(T, dim_in, Tuple(1:N); array_type)
end
function FFTShift(dim_in::NTuple{N, Int}, dirs; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return FFTShift(Float64, dim_in, dirs; array_type)
end
function FFTShift(dim_in::NTuple{N, Int}; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return FFTShift(Float64, dim_in, Tuple(1:N); array_type)
end
FFTShift(dim_in::Vararg{Int}) = FFTShift(dim_in)
function FFTShift(x::A) where {A <: AbstractArray}
    return FFTShift(eltype(x), size(x); array_type = typeof(x isa SubArray ? parent(x) : x))
end

function IFFTShift(
        ::Type{T}, dim_in::NTuple{N, Int}, dirs;
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return _make_shift(IFFTShift, T, dim_in, dirs, array_type)
end
function IFFTShift(
        ::Type{T}, dim_in::NTuple{N, Int};
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return IFFTShift(T, dim_in, Tuple(1:N); array_type)
end
function IFFTShift(dim_in::NTuple{N, Int}, dirs; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return IFFTShift(Float64, dim_in, dirs; array_type)
end
function IFFTShift(dim_in::NTuple{N, Int}; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return IFFTShift(Float64, dim_in, Tuple(1:N); array_type)
end
IFFTShift(dim_in::Vararg{Int}) = IFFTShift(dim_in)
function IFFTShift(x::A) where {A <: AbstractArray}
    return IFFTShift(eltype(x), size(x); array_type = typeof(x isa SubArray ? parent(x) : x))
end

function SignAlternation(
        ::Type{T}, dim_in::NTuple{N, Int}, dirs;
        threaded::Bool = true, array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    d = _normalize_dirs(dim_in, dirs)
    S = _normalize_array_type(array_type, T)
    # Routed through the shared resolver like every other operator: `false` vetoes, `true`
    # enables subject to the policy, which also covers the thread count and GPU storage.
    th = _elementwise_threaded(SignAlternation, threaded, T, dim_in, S)
    return SignAlternation{T, N, length(d), th, S}(dim_in, d)
end
function SignAlternation(
        dim_in::NTuple{N, Int}, dirs;
        threaded::Bool = true, array_type::Type{<:AbstractArray} = Array{Float64},
    ) where {N}
    return SignAlternation(Float64, dim_in, dirs; threaded, array_type)
end
function SignAlternation(x::A, dirs; threaded::Bool = true) where {A <: AbstractArray}
    return SignAlternation(
        eltype(x), size(x), dirs;
        threaded, array_type = typeof(x isa SubArray ? parent(x) : x),
    )
end

domain_type(::FFTShift{T}) where {T} = T
codomain_type(::FFTShift{T}) where {T} = T
domain_array_type(::FFTShift{T, N, M, S}) where {T, N, M, S} = S
codomain_array_type(::FFTShift{T, N, M, S}) where {T, N, M, S} = S
domain_type(::IFFTShift{T}) where {T} = T
codomain_type(::IFFTShift{T}) where {T} = T
domain_array_type(::IFFTShift{T, N, M, S}) where {T, N, M, S} = S
codomain_array_type(::IFFTShift{T, N, M, S}) where {T, N, M, S} = S
domain_type(::SignAlternation{T}) where {T} = T
codomain_type(::SignAlternation{T}) where {T} = T
domain_array_type(::SignAlternation{T, N, M, Th, S}) where {T, N, M, Th, S} = S
codomain_array_type(::SignAlternation{T, N, M, Th, S}) where {T, N, M, Th, S} = S

size(L::FFTShift) = (L.dim_in, L.dim_in)
size(L::IFFTShift) = (L.dim_in, L.dim_in)
size(L::SignAlternation) = (L.dim_in, L.dim_in)

fun_name(::FFTShift) = "⇉"
fun_name(::IFFTShift) = "⇇"
fun_name(::SignAlternation) = "±"

is_thread_safe(::FFTShift) = true
is_thread_safe(::IFFTShift) = true
is_thread_safe(::SignAlternation) = true

is_AcA_diagonal(::FFTShift) = true
is_AAc_diagonal(::FFTShift) = true
diag_AcA(L::FFTShift) = one(real(domain_type(L)))
diag_AAc(L::FFTShift) = one(real(domain_type(L)))
is_symmetric(L::FFTShift) = all(d -> iseven(L.dim_in[d]), L.dirs)
is_orthogonal(::FFTShift) = true
is_invertible(::FFTShift) = true
is_full_row_rank(::FFTShift) = true
is_full_column_rank(::FFTShift) = true

is_AcA_diagonal(::IFFTShift) = true
is_AAc_diagonal(::IFFTShift) = true
diag_AcA(L::IFFTShift) = one(real(domain_type(L)))
diag_AAc(L::IFFTShift) = one(real(domain_type(L)))
is_symmetric(L::IFFTShift) = all(d -> iseven(L.dim_in[d]), L.dirs)
is_orthogonal(::IFFTShift) = true
is_invertible(::IFFTShift) = true
is_full_row_rank(::IFFTShift) = true
is_full_column_rank(::IFFTShift) = true

is_AcA_diagonal(::SignAlternation) = true
is_AAc_diagonal(::SignAlternation) = true
diag_AcA(L::SignAlternation) = one(real(domain_type(L)))
diag_AAc(L::SignAlternation) = one(real(domain_type(L)))
is_symmetric(::SignAlternation) = true
is_orthogonal(::SignAlternation) = true
is_invertible(::SignAlternation) = true
is_full_row_rank(::SignAlternation) = true
is_full_column_rank(::SignAlternation) = true

has_fast_opnorm(::Union{FFTShift, IFFTShift, SignAlternation}) = true
function LinearAlgebra.opnorm(L::Union{FFTShift, IFFTShift, SignAlternation})
    return one(real(domain_type(L)))
end

function mul!(y::AbstractArray, L::FFTShift, b::AbstractArray)
    check(y, L, b)
    return FFTW.fftshift!(y, b, L.dirs)
end

function mul!(y::AbstractArray, L::IFFTShift, b::AbstractArray)
    check(y, L, b)
    return FFTW.ifftshift!(y, b, L.dirs)
end

function mul!(y::AbstractArray, L::AdjointOperator{<:FFTShift}, b::AbstractArray)
    check(y, L, b)
    return FFTW.ifftshift!(y, b, L.A.dirs)
end

function mul!(y::AbstractArray, L::AdjointOperator{<:IFFTShift}, b::AbstractArray)
    check(y, L, b)
    return FFTW.fftshift!(y, b, L.A.dirs)
end

"""
        alternate_sign!(x[, dirs...])
In-place sign alternation across specified dimensions. Flips the sign where the parity sum across `dirs` is odd.
The provided `dirs` must be within `1:ndims(x)`, and sorted in ascending order. They can also be provided as a tuple.

```jldoctest
julia> using FFTWOperators

julia> v = collect(1.0:4.0);

julia> alternate_sign!(v, 1)
4-element Vector{Float64}:
  1.0
 -2.0
  3.0
 -4.0

julia> M = ones(2,2); alternate_sign!(M, 1, 2)
2×2 Matrix{Float64}:
  1.0  -1.0
 -1.0   1.0

julia> M = ones(2,2); alternate_sign!(M, (1, 2))
2×2 Matrix{Float64}:
  1.0  -1.0
 -1.0   1.0
```
"""
function alternate_sign!(x::AbstractArray, dirs::Int...; threaded::Bool = true)
    return alternate_sign!(x, dirs; threaded)
end

function alternate_sign!(
        x::AbstractArray, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {M}
    _check_shift_dirs(size(x), dirs)
    return _alternate_sign!(x, dirs; threaded)
end

function _alternate_sign!(
        x::AbstractArray{<:Any, N}, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {N, M}
    if isempty(dirs)
        return x
    end
    rest_mask = ntuple(k -> (k + 1) in dirs, Val(N - 1))
    rest_range = CartesianIndices(Base.tail(size(x)))
    use_threads = threaded && Threads.nthreads() > 1
    # `in1` decides the shape of the inner loop, so it is lifted to a type parameter here
    # rather than tested per element inside it.
    if 1 in dirs
        _alternate_sign_columns!(x, Val(true), rest_mask, rest_range, use_threads)
    else
        _alternate_sign_columns!(x, Val(false), rest_mask, rest_range, use_threads)
    end
    return x
end

function _alternate_sign_columns!(
        x::AbstractArray, ::Val{IN1}, rest_mask::NTuple{K, Bool},
        rest_range::CartesianIndices, use_threads::Bool
    ) where {IN1, K}
    n = size(x, 1)
    ncol = length(rest_range)
    if use_threads && ncol > 1
        # `Threads.@threads`, not `@batch`, and only here: this loop's body is a call out to
        # `_alternate_sign_column!` rather than straight-line element work, and `Polyester.@batch`
        # does not resolve such a body statically inside a precompiled package — every call takes
        # a dynamic path instead. Measured on a 128x128x8 `ComplexF32` out-of-place pass, AMD
        # EPYC 7352, 8 Julia threads, 2026-09-21: 583.0 us with `@batch` against 56.4 us for the
        # serial loop it was supposed to beat, and 21.0 us here. The single-column `@batch`
        # kernels below are straight-line and keep it; measured on a 2^20 vector they are 4.8x
        # (out of place) and 10.6x (in place) up on serial.
        @inbounds Threads.@threads for c in 1:ncol
            _alternate_sign_column!(x, Val(IN1), rest_mask, rest_range[c], c, n, Val(false))
        end
    elseif use_threads && n > 1
        # A single trailing column (a vector, or an `n×1`): the column loop has nothing to
        # spread across workers, so thread dimension 1 itself rather than run the whole pass
        # sequentially.
        _alternate_sign_column!(x, Val(IN1), rest_mask, first(rest_range), 1, n, Val(true))
    else
        @inbounds for c in 1:ncol
            _alternate_sign_column!(x, Val(IN1), rest_mask, rest_range[c], c, n, Val(false))
        end
    end
    return
end

# Dimension 1's own alternation, and the parity contribution of dimensions 2:N for one column.
# `N` is a static type parameter and the dim-1 sign is a predicate, so neither needs a heap array.
@inline _dim1_sign(in1::Bool, i::Integer) = (in1 && iseven(i)) ? -1 : 1

# Below this many elements, spreading a single column over Polyester workers costs more in
# fork/join than the multiplies it saves — a 256-point readout is a few hundred nanoseconds of
# work against microseconds of setup.
const MIN_ELEMENTS_FOR_COLUMN_THREADING = 4096

@inline function _column_sign(rest_mask::NTuple{K, Bool}, J::CartesianIndex) where {K}
    Jt = Tuple(J)
    rest_flips = 0
    @inbounds for k in 1:K
        if rest_mask[k] && iseven(Jt[k])
            rest_flips += 1
        end
    end
    return isodd(rest_flips) ? -1 : 1
end

# A column of an array that indexes linearly and starts at 1 is the contiguous stretch
# `(c - 1) * size(x, 1) .+ (1:size(x, 1))` of its linear index space — the AbstractArray
# interface guarantees linear indexing runs in column-major order. Addressing it that way
# instead of splatting a `CartesianIndex` into the inner loop is what makes the kernel
# vectorize; the Cartesian form below is the correctness fallback for everything else
# (non-contiguous views, offset axes).
@inline function _has_linear_columns(x::AbstractArray)
    return IndexStyle(x) === IndexLinear() && !Base.has_offset_axes(x)
end

# The parity contribution from dims 2:N is loop-invariant across dim 1, so it is computed once
# per column ("per-slab base parity") and the inner loop over dim 1 — the only dimension that
# can alternate every element — vectorizes with `@simd`. `Val(true)` spreads that inner loop
# over workers instead, for the caller that has only one column to work with; the branch is on a
# type parameter, so the unused loop is compiled away.
@inline function _alternate_sign_column!(
        x::AbstractArray, ::Val{IN1}, rest_mask::NTuple{K, Bool}, J::CartesianIndex,
        c::Integer, n::Integer, ::Val{TH} = Val(false)
    ) where {IN1, K, TH}
    neg = _column_sign(rest_mask, J) < 0
    if _has_linear_columns(x)
        _alt_column_linear!(x, Val(IN1), neg, (c - 1) * n, n, Val(TH))
    else
        _alt_column_cartesian!(x, Val(IN1), neg, Tuple(J), Val(TH))
    end
    return
end

# In place, every sign is ±1 and the operation is its own inverse, so only the elements that
# actually flip need to be touched at all: a strided half pass of negations, not a full pass of
# multiplications. That is worth 1.7-8.9x over the full pass, measured; see
# `threading_threshold(::Type{<:SignAlternation})`.
@inline function _alt_column_linear!(
        x::AbstractArray, ::Val{IN1}, neg::Bool, base::Integer, n::Integer, ::Val{TH}
    ) where {IN1, TH}
    if IN1
        # `neg` flips the whole column, so the elements left standing are the even ones;
        # without it the alternation itself flips the even ones.
        start = neg ? 1 : 2
        if TH
            @inbounds @batch minbatch = MIN_ELEMENTS_FOR_COLUMN_THREADING for i in start:2:n
                x[base + i] = -x[base + i]
            end
        else
            @inbounds @simd for i in start:2:n
                x[base + i] = -x[base + i]
            end
        end
    elseif neg
        if TH
            @inbounds @batch minbatch = MIN_ELEMENTS_FOR_COLUMN_THREADING for i in 1:n
                x[base + i] = -x[base + i]
            end
        else
            @inbounds @simd for i in 1:n
                x[base + i] = -x[base + i]
            end
        end
    end
    return
end

@inline function _alt_column_cartesian!(
        x::AbstractArray, ::Val{IN1}, neg::Bool, Jt::Tuple, ::Val{TH}
    ) where {IN1, TH}
    ax = axes(x, 1)
    f, l = first(ax), last(ax)
    if IN1
        start = (isodd(f) == neg) ? f : f + 1
        if TH
            @inbounds @batch minbatch = MIN_ELEMENTS_FOR_COLUMN_THREADING for i in start:2:l
                x[i, Jt...] = -x[i, Jt...]
            end
        else
            @inbounds @simd for i in start:2:l
                x[i, Jt...] = -x[i, Jt...]
            end
        end
    elseif neg
        if TH
            @inbounds @batch minbatch = MIN_ELEMENTS_FOR_COLUMN_THREADING for i in ax
                x[i, Jt...] = -x[i, Jt...]
            end
        else
            @inbounds @simd for i in ax
                x[i, Jt...] = -x[i, Jt...]
            end
        end
    end
    return
end

"""
    alternate_sign!(y, x[, dirs...])
Out-of-place variant: apply sign alternation of `x` across `dirs` and store the result in `y`.
`y` and `x` must have the same size. The provided `dirs` must be within `1:ndims(x)`, and
sorted in ascending order. They can also be provided as a tuple.

```jldoctest
julia> using FFTWOperators

julia> x = reshape(1.0:4.0, 2, 2); y = similar(x);

julia> alternate_sign!(y, x, 1, 2)
2×2 Matrix{Float64}:
  1.0  -3.0
 -2.0   4.0

julia> alternate_sign!(y, x, (1, 2))
2×2 Matrix{Float64}:
  1.0  -3.0
 -2.0   4.0
```
"""
function alternate_sign!(
        y::AbstractArray, x::AbstractArray, dirs::Int...; threaded::Bool = true
    )
    return alternate_sign!(y, x, dirs; threaded)
end
function alternate_sign!(
        y::AbstractArray, x::AbstractArray, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {M}
    _check_shift_dirs(size(x), dirs)
    return _alternate_sign!(y, x, dirs; threaded)
end

function _alternate_sign!(
        y::AbstractArray{<:Any, N}, x::AbstractArray{<:Any, N}, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {N, M}
    size(y) == size(x) || throw(ArgumentError("y and x must have the same size"))
    if isempty(dirs)
        y .= x
        return y
    end
    rest_mask = ntuple(k -> (k + 1) in dirs, Val(N - 1))
    rest_range = CartesianIndices(Base.tail(size(x)))
    use_threads = threaded && Threads.nthreads() > 1
    if 1 in dirs
        _alternate_sign_columns!(y, x, Val(true), rest_mask, rest_range, use_threads)
    else
        _alternate_sign_columns!(y, x, Val(false), rest_mask, rest_range, use_threads)
    end
    return y
end

function _alternate_sign_columns!(
        y::AbstractArray, x::AbstractArray, ::Val{IN1}, rest_mask::NTuple{K, Bool},
        rest_range::CartesianIndices, use_threads::Bool
    ) where {IN1, K}
    n = size(x, 1)
    ncol = length(rest_range)
    if use_threads && ncol > 1
        # See the in-place variant: `@batch` mis-compiles this call-out body in a precompiled
        # package and loses to the serial loop by an order of magnitude.
        @inbounds Threads.@threads for c in 1:ncol
            _alternate_sign_column!(y, x, Val(IN1), rest_mask, rest_range[c], c, n, Val(false))
        end
    elseif use_threads && n > 1
        # See the in-place variant: a single trailing column leaves the column loop with nothing
        # to spread, so thread dimension 1 instead.
        _alternate_sign_column!(y, x, Val(IN1), rest_mask, first(rest_range), 1, n, Val(true))
    else
        @inbounds for c in 1:ncol
            _alternate_sign_column!(y, x, Val(IN1), rest_mask, rest_range[c], c, n, Val(false))
        end
    end
    return
end

@inline function _alternate_sign_column!(
        y::AbstractArray, x::AbstractArray, ::Val{IN1}, rest_mask::NTuple{K, Bool},
        J::CartesianIndex, c::Integer, n::Integer, ::Val{TH} = Val(false)
    ) where {IN1, K, TH}
    neg = _column_sign(rest_mask, J) < 0
    if _has_linear_columns(y) && _has_linear_columns(x)
        _alt_column_linear!(y, x, Val(IN1), neg, (c - 1) * n, n, Val(TH))
    else
        _alt_column_cartesian!(y, x, Val(IN1), neg, Tuple(J), Val(TH))
    end
    return
end

# Out of place every element of `y` must be written whatever its sign, so — unlike the in-place
# kernel — there is no half pass to be had and the loop stays contiguous, the sign coming from a
# branchless select. Against the previous splatted-`CartesianIndex` loop this is 6.4x on the
# dynamic case, serial (measured, AMD EPYC 7352, ComplexF32, 2026-09-20).
@inline function _alt_column_linear!(
        y::AbstractArray, x::AbstractArray, ::Val{IN1}, neg::Bool,
        base::Integer, n::Integer, ::Val{TH}
    ) where {IN1, TH}
    if IN1
        s = neg ? -1 : 1
        if TH
            @inbounds @batch minbatch = MIN_ELEMENTS_FOR_COLUMN_THREADING for i in 1:n
                y[base + i] = ifelse(isodd(i), s, -s) * x[base + i]
            end
        else
            @inbounds @simd for i in 1:n
                y[base + i] = ifelse(isodd(i), s, -s) * x[base + i]
            end
        end
    else
        s = neg ? -1 : 1
        if TH
            @inbounds @batch minbatch = MIN_ELEMENTS_FOR_COLUMN_THREADING for i in 1:n
                y[base + i] = s * x[base + i]
            end
        else
            @inbounds @simd for i in 1:n
                y[base + i] = s * x[base + i]
            end
        end
    end
    return
end

@inline function _alt_column_cartesian!(
        y::AbstractArray, x::AbstractArray, ::Val{IN1}, neg::Bool, Jt::Tuple, ::Val{TH}
    ) where {IN1, TH}
    column_sign = neg ? -1 : 1
    if TH
        @inbounds @batch minbatch = MIN_ELEMENTS_FOR_COLUMN_THREADING for i in axes(x, 1)
            y[i, Jt...] = column_sign * _dim1_sign(IN1, i) * x[i, Jt...]
        end
    else
        @inbounds @simd for i in axes(x, 1)
            y[i, Jt...] = column_sign * _dim1_sign(IN1, i) * x[i, Jt...]
        end
    end
    return
end

"""
    alternate_sign(x[, dirs...])
Returns a copy with sign alternation across specified dimensions.

```jldoctest
julia> using FFTWOperators

julia> alternate_sign(collect(1.0:4.0), 1)
4-element Vector{Float64}:
  1.0
 -2.0
  3.0
 -4.0
```
"""
function alternate_sign(x::AbstractArray, dirs::Int...; threaded::Bool = true)
    return alternate_sign!(copy(x), dirs...; threaded)
end

function mul!(
        y::AbstractArray, L::SignAlternation{T, N, M, Th}, b::AbstractArray
    ) where {T, N, M, Th}
    check(y, L, b)
    # In place the kernel only has to touch the elements that flip, so an aliased call is
    # worth routing to the in-place variant rather than copying each element onto itself.
    y === b && return _alternate_sign!(y, L.dirs; threaded = Th)
    return _alternate_sign!(y, b, L.dirs; threaded = Th)
end

has_optimized_normalop(::Union{FFTShift, IFFTShift, SignAlternation}) = true
function get_normal_op(L::Union{FFTShift, IFFTShift, SignAlternation})
    return Eye(domain_type(L), size(L, 1); array_type = domain_array_type(L))
end

LinearAlgebra.adjoint(L::SignAlternation) = L

# In a `Compose`, a sign alternation is a pointwise map: the sign of element `j` follows from
# its linear index.
AbstractOperators._pw_kind(::Type{<:SignAlternation{T, N, M, Th, S}}) where {T, N, M, Th, S <: Array} =
    AbstractOperators.PwMapKind()

function AbstractOperators._pw_steps(L::SignAlternation{T}) where {T}
    strides = Base.size_to_strides(1, L.dim_in...)
    return (PwSignAlternation{T}(map(d -> strides[d], L.dirs), map(d -> L.dim_in[d], L.dirs), prod(L.dim_in)),)
end

# -v where the indices along the dimensions with strides `strides` and sizes `sizes` have an odd
# sum of (zero-based) parities, v elsewhere.
struct PwSignAlternation{T, M} <: AbstractOperators.PwStep{T}
    strides::NTuple{M, Int}
    sizes::NTuple{M, Int}
    len::Int
end
PwSignAlternation{T}(strides::NTuple{M, Int}, sizes::NTuple{M, Int}, len::Int) where {T, M} =
    PwSignAlternation{T, M}(strides, sizes, len)

AbstractOperators._pw_length(s::PwSignAlternation) = s.len

_pw_flip(s::PwSignAlternation, j) =
    isodd(sum(map((st, n) -> rem(div(j - 1, st), n), s.strides, s.sizes)))

@inline AbstractOperators._pw_apply(s::PwSignAlternation{T}, v, j) where {T} =
    convert(T, ifelse(_pw_flip(s, j), -v, v))

# The CPU kernels work in segments that start at a multiple of the period and do not cross one,
# so on a segment the parity of every dimension but the first is constant. The first dimension
# (stride 1) alternates along the segment: across the end of a column too when its size is
# even, which is why only an odd first size sets the period.
function AbstractOperators._pw_period(s::PwSignAlternation)
    p = nothing
    for (st, n) in zip(s.strides, s.sizes)
        q = st == 1 ? (isodd(n) ? n : nothing) : st
        q === nothing || (p = p === nothing ? q : min(p, q))
    end
    return p
end

function AbstractOperators._pw_at(s::PwSignAlternation{T}, j0) where {T}
    return PwSignSegment{T}(_pw_flip(s, j0), any(==(1), s.strides), j0)
end

# The sign alternation resolved for the segment that starts at element `j0`.
struct PwSignSegment{T} <: AbstractOperators.PwStep{T}
    flip0::Bool
    alternate::Bool
    j0::Int
end

@inline function AbstractOperators._pw_apply(s::PwSignSegment{T}, v, j) where {T}
    flip = s.flip0 ⊻ (s.alternate & isodd(j - s.j0))
    return convert(T, ifelse(flip, -v, v))
end

# Utility

function _check_shift_dirs(::NTuple{N, Int}, dirs::NTuple{M, Int}) where {N, M}
    if M == 0
        throw(ArgumentError("dirs must contain at least one dimension"))
    end
    if N < M
        throw(ArgumentError("Number of dirs exceeds number of dimensions of x"))
    end
    if dirs[1] < 1 || dirs[end] > N
        throw(ArgumentError("dirs must be in 1:$N"))
    end
    for i in 2:M
        if dirs[i] < dirs[i - 1]
            throw(ArgumentError("dirs must be sorted in ascending order"))
        end
    end
    return nothing
end

function _is_dft_op(op, side)
    if op isa DFT || op isa AdjointOperator{<:DFT}
        return true
    elseif op isa Compose
        subops = AbstractOperators.get_operators(op)
        if side == :domain
            # Domain shift: innermost (first) op must be DFT-like; all outer ops must be
            # diagonal so they commute with SignAlternation.
            return _is_dft_op(first(subops), side) && all(is_diagonal, subops[2:end])
        else
            # Codomain shift: outermost (last) op must be DFT-like; all inner ops must be
            # diagonal so they commute with SignAlternation.
            return _is_dft_op(last(subops), side) && all(is_diagonal, subops[1:(end - 1)])
        end
    else
        return false
    end
end

# ─── Shifts absorbed into a DFT as phase ramps ────────────────────────────────
#
# A circular shift by s along a dimension of length N next to a transform with kernel
# exp(σ·2πi·n·k/N) moves to the other side of the transform as a diagonal phase ramp:
#   F ∘ S_s = diag(exp(σ·2πi·s·k/N)) ∘ F        (shift of the input; ramp over the output index k)
#   S_s ∘ F = F ∘ diag(exp(-σ·2πi·s·n/N))       (shift of the output; ramp over the input index n)
# with σ = -1 for `DFT` and σ = +1 for its adjoint. For even N and s = ±N/2 the ramp is (-1)^k,
# which `SignAlternation` applies without storing it; odd N needs the complex ramp, stored as a
# `DiagOp` over the whole array. A ramp over the input of the transform is complex, so it
# needs a complex domain.

_dft_sign(::DFT) = -1
_dft_sign(::AdjointOperator{<:DFT}) = 1

_dft_dirs(::DFT{N, C, D, Dir}) where {N, C, D, Dir} = Dir
_dft_dirs(op::AdjointOperator{<:DFT}) = _dft_dirs(op.A)

_shift_dirs(S::ShiftOp) = S.dirs
_shift_dirs(S::AdjointOperator{<:ShiftOp}) = S.A.dirs

# Circular shift of each shift operator along a dimension of length `n`, as `circshift` counts it.
_shift_amount(::Type{<:FFTShift}, n::Integer) = n ÷ 2
_shift_amount(::Type{<:IFFTShift}, n::Integer) = -(n ÷ 2)
_shift_amount(S::ShiftOp, n::Integer) = _shift_amount(typeof(S), n)
_shift_amount(S::AdjointOperator{<:ShiftOp}, n::Integer) = -_shift_amount(S.A, n)

# The DFT-like operator next to a shift on `side` of `op`, which `_is_dft_op(op, side)` accepted.
function _dft_part(op, side)
    op isa Compose || return op
    subops = AbstractOperators.get_operators(op)
    return side == :domain ? first(subops) : last(subops)
end

# `exp(σ·2πi·s·(k - offset)/n)` for k = 0:n-1; the argument is reduced mod n before scaling.
_ramp_vector(n, s, σ, offset) = [cispi(2 * σ * mod(s * (k - offset), n) / n) for k in 0:(n - 1)]

# The phase ramp along `dirs` as a `DiagOp` on an array like `like` (allocated with its type and
# size). `amount(n)` is the shift along a dimension of length `n`, `offsets[i]` an index offset
# along `dirs[i]`. A ramp standing in for a shift threads exactly when the operator it sits
# next to does (`threaded`), so a transform built with `threaded = false` stays serial.
function _phase_ramp_op(like::AbstractArray, dirs, amount, σ, offsets = map(_ -> 0, dirs); threaded::Bool)
    host = ones(eltype(like), size(like))
    for (dir, offset) in zip(dirs, offsets)
        n = size(like, dir)
        v = _ramp_vector(n, amount(n), σ, offset)
        host .*= reshape(v, ntuple(i -> i == dir ? n : 1, ndims(like)))
    end
    d = copyto!(like, host)
    return DiagOp(eltype(like), size(like), d; threaded)
end

# Whether a shift by `dirs` on `side` of the DFT-like `F` can become a ramp on its other side.
function _ramp_absorbs(F, dirs, side)
    all(in(_dft_dirs(F)), dirs) || return false
    side == :codomain && !(domain_type(F) <: Complex) && return false
    return true
end

_all_even(dims::Tuple, dirs) = all(iseven, dims[collect(dirs)])

# A shift of odd length on both sides of a bare DFT: S_a ∘ F ∘ S_b = (c·D_b) ∘ F ∘ M_a, where the
# codomain shift moved past the ramp D_b multiplies it by c = exp(-σ·2πi·a·b/N) in every
# dimension shifted on both sides; that is D_b evaluated at k - a.
function _shift_both_sides(shift_op_type, F, domain_shifts, codomain_shifts)
    σ = _dft_sign(F)
    amount = n -> _shift_amount(shift_op_type, n)
    offsets = map(d -> d in codomain_shifts ? amount(size(F, 1)[d]) : 0, domain_shifts)
    threaded = is_threaded(F)
    D = _phase_ramp_op(allocate_in_codomain(F), domain_shifts, amount, σ, offsets; threaded)
    M = _phase_ramp_op(allocate_in_domain(F), codomain_shifts, amount, -σ; threaded)
    return D * F * M
end

# A sign alternation or phase ramp standing in for a shift threads exactly when the operator it
# wraps does, so a transform built with `threaded = false` stays serial once it is shifted.
function _shift_op(
        shift_op_type, op::AbstractOperator, domain_shifts::Tuple = (), codomain_shifts::Tuple = ()
    )
    threaded = is_threaded(op)
    if !isempty(domain_shifts) && !isempty(codomain_shifts) &&
            (op isa DFT || op isa AdjointOperator{<:DFT}) &&
            !(_all_even(size(op, 2), domain_shifts) && _all_even(size(op, 1), codomain_shifts))
        _check_shift_dirs(size(op, 2), domain_shifts)
        _check_shift_dirs(size(op, 1), codomain_shifts)
        if _ramp_absorbs(op, domain_shifts, :domain) && _ramp_absorbs(op, codomain_shifts, :codomain)
            return _shift_both_sides(shift_op_type, op, domain_shifts, codomain_shifts)
        end
    end
    if !isempty(domain_shifts)
        _check_shift_dirs(size(op, 2), domain_shifts)
        shifted_domain_dims_shape = size(op, 2)[collect(domain_shifts)]
        if all(iseven, shifted_domain_dims_shape) && _is_dft_op(op, :domain) &&
                all(in(_dft_dirs(_dft_part(op, :domain))), domain_shifts)
            domain_op = SignAlternation(codomain_type(op), size(op, 1), domain_shifts; threaded, array_type = codomain_array_type(op))
            op = domain_op * op
        elseif _is_dft_op(op, :domain) && _ramp_absorbs(_dft_part(op, :domain), domain_shifts, :domain)
            σ = _dft_sign(_dft_part(op, :domain))
            op = _phase_ramp_op(allocate_in_codomain(op), domain_shifts, n -> _shift_amount(shift_op_type, n), σ; threaded) * op
        else
            domain_op = shift_op_type(domain_type(op), size(op, 2), domain_shifts; array_type = domain_array_type(op))
            op = op * domain_op
        end
    end
    if !isempty(codomain_shifts)
        _check_shift_dirs(size(op, 1), codomain_shifts)
        shifted_codomain_dims_shape = size(op, 1)[collect(codomain_shifts)]
        if all(iseven, shifted_codomain_dims_shape) && _is_dft_op(op, :codomain) &&
                all(in(_dft_dirs(_dft_part(op, :codomain))), codomain_shifts)
            codomain_op = SignAlternation(domain_type(op), size(op, 2), codomain_shifts; threaded, array_type = domain_array_type(op))
            op = op * codomain_op
        elseif _is_dft_op(op, :codomain) && _ramp_absorbs(_dft_part(op, :codomain), codomain_shifts, :codomain)
            σ = _dft_sign(_dft_part(op, :codomain))
            op = op * _phase_ramp_op(allocate_in_domain(op), codomain_shifts, n -> _shift_amount(shift_op_type, n), -σ; threaded)
        else
            codomain_op = shift_op_type(codomain_type(op), size(op, 1), codomain_shifts; array_type = codomain_array_type(op))
            op = codomain_op * op
        end
    end
    return op
end

"""
    fftshift_op(op::AbstractOperator; domain_shifts::Tuple=(), codomain_shifts::Tuple=())

Applies `FFTShift` to the domain and/or codomain of `op`, depending on `domain_shifts` and `codomain_shifts`.
If `op` is a (possibly modified) DFT/IDFT that transforms the shifted dimensions, the shift is replaced by a
phase ramp on the other side of the DFT/IDFT: a `SignAlternation` when every shifted dimension has even length,
otherwise a `DiagOp` holding the complex ramp. A ramp on the input side needs a complex domain; for a real one
the shift is kept.

# Examples
```jldoctest
julia> using FFTWOperators, FFTW

julia> x = rand(15);

julia> F = fftshift_op(DFT(15); domain_shifts=(1,)) # 15 is odd, so the shift becomes a phase ramp (╲)
╲*ℱ  ℝ^15 -> ℂ^15

julia> F * x ≈ FFTW.fft(FFTW.fftshift(x))
true

julia> F = fftshift_op(DFT(15); codomain_shifts=(1,))
⇉*ℱ  ℝ^15 -> ℂ^15

julia> F * x ≈ FFTW.fftshift(FFTW.fft(x))
true

julia> F = fftshift_op(DFT(15); domain_shifts=(1,), codomain_shifts=(1,))
Π  ℝ^15 -> ℂ^15

julia> F * x ≈ FFTW.fftshift(FFTW.fft(FFTW.fftshift(x)))
true

julia> F = fftshift_op(DFT(16); codomain_shifts=(1,)) # note that 16 is even, so we get a SignAlternation (±)
ℱ*±  ℝ^16 -> ℂ^16


```
"""
function fftshift_op(
        op::AbstractOperator; domain_shifts::Tuple = (), codomain_shifts::Tuple = ()
    )
    return _shift_op(FFTShift, op, domain_shifts, codomain_shifts)
end

"""
    ifftshift_op(op::AbstractOperator; domain_shifts::Tuple=(), codomain_shifts::Tuple=())

Applies `IFFTShift` to the domain and/or codomain of `op`, depending on `domain_shifts` and `codomain_shifts`.
If `op` is a (possibly modified) DFT/IDFT that transforms the shifted dimensions, the shift is replaced by a
phase ramp on the other side of the DFT/IDFT: a `SignAlternation` when every shifted dimension has even length,
otherwise a `DiagOp` holding the complex ramp. A ramp on the input side needs a complex domain; for a real one
the shift is kept.

# Examples
```jldoctest
julia> using FFTWOperators, FFTW

julia> x = rand(ComplexF64, 15);

julia> F = ifftshift_op(IDFT(15); domain_shifts=(1,))
╲*ℱ⁻¹  ℂ^15 -> ℂ^15

julia> F * x ≈ FFTW.ifft(FFTW.ifftshift(x))
true

julia> F = ifftshift_op(IDFT(15); codomain_shifts=(1,))
ℱ⁻¹*╲  ℂ^15 -> ℂ^15

julia> F * x ≈ FFTW.ifftshift(FFTW.ifft(x))
true

julia> F = ifftshift_op(IDFT(15); domain_shifts=(1,), codomain_shifts=(1,))
Π  ℂ^15 -> ℂ^15

julia> F * x ≈ FFTW.ifftshift(FFTW.ifft(FFTW.ifftshift(x)))
true

julia> F = ifftshift_op(IDFT(16); codomain_shifts=(1,)) # note that 16 is even, so we get a SignAlternation (±)
ℱ⁻¹*±  ℂ^16 -> ℂ^16

```
"""
function ifftshift_op(
        op::AbstractOperator; domain_shifts::Tuple = (), codomain_shifts::Tuple = ()
    )
    return _shift_op(IFFTShift, op, domain_shifts, codomain_shifts)
end

"""
	threading_threshold(::Type{<:SignAlternation})

PROVENANCE: measured (AMD EPYC 7352, 8 Julia threads, OPENBLAS_NUM_THREADS=1, ComplexF32,
`@belapsed` minimum, 2026-09-21), sweeping serial against threaded over powers of two. Ratios
are serial/threaded, so above 1 threading pays:

| elements | vector, `dirs = (1,)` | `N x N x 8`, `dirs = (1, 2)` | `dirs = (1,)` | `dirs = (2,)` |
|---|---|---|---|---|
| 2^12 | 1.06x | 0.18x | 0.20x | 0.09x |
| 2^13 | 0.72x | 0.25x | 0.28x | 0.12x |
| 2^14 | 1.29x | 0.72x | 0.65x | 0.30x |
| 2^15 | 2.68x | 1.12x | 1.06x | 0.49x |
| 2^16 | 3.50x | 1.84x | 1.86x | 1.03x |
| 2^17 | 3.72x | 2.81x | 2.78x | 1.75x |
| 2^18 | 3.90x | 3.49x | 3.40x | 2.48x |

2^16 is the first size that pays across *every* shape and `dirs`, so it is the one the policy
uses. `dirs = (2,)` sets it: alternating a dimension other than the first makes whole columns
uniform, so the kernel's per-column work is a plain scale and there is less of it to spread.

The earlier revision of this table was measured on one-dimensional inputs only, and put the
threshold at 2^14 on 4.15x at that size. A vector is the one shape that never enters the
multi-column loop -- it takes the single-column `@batch` kernels instead -- so the sweep never
exercised the path that carries every real array, and did not see that that path was slower
threaded than serial at *any* size. The columns above marked `N x N x 8` are that path.

The shared `THRESHOLD_MEMORY_BOUND` this used to return is 2^18 -- four times too high -- which
switched threading off for whole classes of real problem: MRT's dynamic encoding operator is
2^17 elements and was running the alternation on one thread.
"""
threading_threshold(::Type{<:SignAlternation}) = 2^16
is_threaded(::SignAlternation{T, N, M, Th}) where {T, N, M, Th} = Th
supports_threading(::SignAlternation) = true

function _copy_operator_impl(
        op::SignAlternation{T, N, M, Th, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, M, Th, S}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return SignAlternation(T, op.dim_in, op.dirs; threaded = new_threaded, array_type = new_at)
end

# FFTShift/IFFTShift have no threaded path of their own.
is_threaded(::ShiftOp) = false
supports_threading(::ShiftOp) = false

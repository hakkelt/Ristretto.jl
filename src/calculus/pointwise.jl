# Fused execution of pointwise runs inside a `Compose`.
#
# A chain such as `DiagOp' * ℱ⁻¹ * ... * DiagOp * BroadCast` spends most of its time moving
# memory: each pointwise operator reads and writes a whole array for a handful of flops. When
# several of them follow one another, `Compose` runs them as one loop instead, which reads the
# run's input once and writes its output once.
#
# An operator opts in by answering `_pw_kind` from its type:
#
#   PwMapKind()       y[j] = f(x[j], j), same shape; `_pw_steps(L)` returns the steps
#   PwExpandKind()    y is copies of x along a group of adjacent axes, laid out as
#                     `_pw_layout(L)` says
#   PwReduceKind()    y is the sum of x over such copies
#   PwEpilogueKind()  `mul!` is a core that is not pointwise followed by a pointwise pass:
#                     `_pw_core!(y, L, x)` runs the core alone, `_pw_epilogue(L)` returns the
#                     pass as one step, and that step joins the maps that follow
#
# A fused run is `Expand Map*`, `Map+ Reduce?` or `Epilogue Map* Reduce?`, taken greedily from
# the left. The kinds are answered from types so that the grouping is resolved at compile time;
# it is done by dispatch rather than inside the generated `mul!` of `Compose`, because a
# generator only sees the methods that existed when it was defined, and packages built on this
# one add opt-ins later.
#
# A fused run computes exactly what the operators compute one after another: every step
# converts its result to the element type of the buffer it would have been written to, and the
# reduction adds the blocks in the same order. The result is therefore the same bit for bit.
#
# Fusion needs storage a kernel can index, and is checked again when the run executes: a run
# whose arrays are neither `Array`s nor arrays a device backend fuses (see `ext/GpuExt`), or
# whose steps do not match the length they are applied to, falls back to the operators' own
# `mul!`s.

abstract type PwKind end
struct PwNoneKind <: PwKind end
struct PwMapKind <: PwKind end
struct PwExpandKind <: PwKind end
struct PwReduceKind <: PwKind end
struct PwEpilogueKind <: PwKind end

_pw_kind(::Type) = PwNoneKind()
_pw_kind(L::AbstractOperator) = _pw_kind(typeof(L))

"""
	_pw_steps(L) -> Tuple

The pointwise steps of an operator whose `_pw_kind` is `PwMapKind()`, in application order.
"""
function _pw_steps end

"""
	_pw_layout(L) -> Union{Tuple{Int, Int}, Nothing}

`(inner, K)` for a `PwExpandKind()` operator that writes, or a `PwReduceKind()` operator that
sums, `K` copies along adjacent axes preceded by `inner` elements; `nothing` when the copies do
not have that form, and the run falls back to the operators' own `mul!`s.
"""
function _pw_layout end

"""
	_pw_core!(y, L, x)

The part of `mul!(y, L, x)` that comes before the pass `_pw_epilogue(L)` describes.
"""
function _pw_core! end

"""
	_pw_epilogue(L) -> step

The pointwise pass that `mul!(y, L, x)` applies after `_pw_core!(y, L, x)`.
"""
function _pw_epilogue end

# ─── Steps ───────────────────────────────────────────────────────────────────
#
# A step maps one element `v`, at linear index `j` of the array it acts on, to an element of
# type `T`. `_pw_length(step)` is the array length the step requires, or `nothing` if any.

abstract type PwStep{T} end

_pw_coef(d::Number, j) = d
Base.@propagate_inbounds _pw_coef(d::AbstractArray, j) = d[j]
_pw_coef_length(::Number) = nothing
_pw_coef_length(d::AbstractArray) = length(d)

# d[j] * v
struct PwLeftMul{T, D} <: PwStep{T}
    d::D
end
PwLeftMul{T}(d::D) where {T, D} = PwLeftMul{T, D}(d)
Base.@propagate_inbounds _pw_apply(s::PwLeftMul{T}, v, j) where {T} = convert(T, _pw_coef(s.d, j) * v)
_pw_length(s::PwLeftMul) = _pw_coef_length(s.d)

# conj(d[j]) * v
struct PwConjLeftMul{T, D} <: PwStep{T}
    d::D
end
PwConjLeftMul{T}(d::D) where {T, D} = PwConjLeftMul{T, D}(d)
Base.@propagate_inbounds _pw_apply(s::PwConjLeftMul{T}, v, j) where {T} =
    convert(T, conj(_pw_coef(s.d, j)) * v)
_pw_length(s::PwConjLeftMul) = _pw_coef_length(s.d)

# real(conj(d[j]) * v)
struct PwRealConjLeftMul{T, D} <: PwStep{T}
    d::D
end
PwRealConjLeftMul{T}(d::D) where {T, D} = PwRealConjLeftMul{T, D}(d)
Base.@propagate_inbounds _pw_apply(s::PwRealConjLeftMul{T}, v, j) where {T} =
    convert(T, real(conj(_pw_coef(s.d, j)) * v))
_pw_length(s::PwRealConjLeftMul) = _pw_coef_length(s.d)

# v * c
struct PwRightMul{T, C <: Number} <: PwStep{T}
    c::C
end
PwRightMul{T}(c::C) where {T, C} = PwRightMul{T, C}(c)
@inline _pw_apply(s::PwRightMul{T}, v, j) where {T} = convert(T, v * s.c)
_pw_length(::PwRightMul) = nothing

# v / c
struct PwDiv{T, C <: Number} <: PwStep{T}
    c::C
end
PwDiv{T}(c::C) where {T, C} = PwDiv{T, C}(c)
@inline _pw_apply(s::PwDiv{T}, v, j) where {T} = convert(T, v / s.c)
_pw_length(::PwDiv) = nothing

# v, unchanged
struct PwIdentity{T} <: PwStep{T} end
@inline _pw_apply(::PwIdentity{T}, v, j) where {T} = convert(T, v)
_pw_length(::PwIdentity) = nothing

# m[(j - 1) ÷ inner + 1] ? v : 0, a mask over the trailing dimensions of the array. It is
# constant over each run of `inner` elements, which the kernels exploit: they work in segments
# that never cross such a run, and `_pw_at` resolves the mask once per segment.
struct PwTrailingMask{T, M <: AbstractArray{Bool}} <: PwStep{T}
    m::M
    inner::Int
end
PwTrailingMask{T}(m::M, inner::Int) where {T, M} = PwTrailingMask{T, M}(m, inner)
_pw_length(s::PwTrailingMask) = s.inner * length(s.m)

# m[j] ? v : 0, a mask over the whole array
struct PwMask{T, M <: AbstractArray{Bool}} <: PwStep{T}
    m::M
end
PwMask{T}(m::M) where {T, M} = PwMask{T, M}(m)
Base.@propagate_inbounds _pw_apply(s::PwMask{T}, v, j) where {T} = ifelse(s.m[j], convert(T, v), zero(T))
_pw_length(s::PwMask) = length(s.m)

# v, or 0: the mask resolved for one segment
struct PwKeep{T} <: PwStep{T}
    keep::Bool
end
@inline _pw_apply(s::PwKeep{T}, v, j) where {T} = ifelse(s.keep, convert(T, v), zero(T))

# The step to apply on the segment starting at element `j0`, and the length of the runs over
# which a step is constant (`nothing` if it has none).
_pw_at(s::PwStep, j0) = s
Base.@propagate_inbounds _pw_at(s::PwTrailingMask{T}, j0) where {T} = PwKeep{T}(s.m[div(j0 - 1, s.inner) + 1])
_pw_period(::PwStep) = nothing
_pw_period(s::PwTrailingMask) = s.inner

@inline _pw_apply_all(::Tuple{}, v, j) = v
Base.@propagate_inbounds _pw_apply_all(steps::Tuple, v, j) =
    _pw_apply_all(Base.tail(steps), _pw_apply(first(steps), v, j), j)

Base.@propagate_inbounds _pw_at_all(steps::Tuple, j0) = map(s -> _pw_at(s, j0), steps)

_pw_steps_fit(::Tuple{}, n::Int) = true
function _pw_steps_fit(steps::Tuple, n::Int)
    len = _pw_length(first(steps))
    return (len === nothing || len == n) && _pw_steps_fit(Base.tail(steps), n)
end

# ─── Kernels ─────────────────────────────────────────────────────────────────
#
# One loop for the three shapes a run can have. An expansion or a reduction has the layout
# `(inner, K)`: the full array is `(inner, K, outer)` and the other side `(inner, outer)`, so
# copy `k` of element `i = ii + (o - 1) * inner` of the small side is element
# `i + (o - 1) * inner * (K - 1) + (k - 1) * inner` of the full one. A plain run is the case
# `K = 1`, `inner` = the whole array.
#
# Each row of `inner` elements is cut into segments of at most `seg` elements, which are the
# unit of work. Within a segment every copy is contiguous in the full array. When a step has a
# period, `seg` divides both it and `inner`, so a segment also stays inside one run of the step.

struct PwPlain end
struct PwExpand end
struct PwReduce end

# Segment length when no step constrains it: long enough to amortize resolving the steps, short
# enough that a reduction's partial sums stay in the L1 cache.
const PW_SEGMENT = 2048

_pw_segment_length(steps::Tuple, inner::Int) = _pw_segment_length(steps, inner, 0)
_pw_segment_length(::Tuple{}, inner::Int, g::Int) = g == 0 ? min(inner, PW_SEGMENT) : g
function _pw_segment_length(steps::Tuple, inner::Int, g::Int)
    p = _pw_period(first(steps))
    g = p === nothing ? g : gcd(g == 0 ? inner : g, p)
    return _pw_segment_length(Base.tail(steps), inner, g)
end

Base.@propagate_inbounds function _pw_segment!(y, x, steps, ::PwPlain, inner, K, lo, hi, base)
    s = _pw_at_all(steps, lo)
    @simd ivdep for j in lo:hi
        y[j] = _pw_apply_all(s, x[j], j)
    end
    return nothing
end

Base.@propagate_inbounds function _pw_segment!(y, x, steps, ::PwExpand, inner, K, lo, hi, base)
    for k in 1:K
        off = base + (k - 1) * inner
        s = _pw_at_all(steps, lo + off)
        @simd ivdep for i in lo:hi
            y[i + off] = _pw_apply_all(s, x[i], i + off)
        end
    end
    return nothing
end

# The copies are added in order, starting from zero, as `sum!` does.
Base.@propagate_inbounds function _pw_segment!(y, x, steps, ::PwReduce, inner, K, lo, hi, base)
    s = _pw_at_all(steps, lo + base)
    @simd ivdep for i in lo:hi
        y[i] = zero(eltype(y)) + _pw_apply_all(s, x[i + base], i + base)
    end
    for k in 2:K
        off = base + (k - 1) * inner
        s = _pw_at_all(steps, lo + off)
        @simd ivdep for i in lo:hi
            y[i] += _pw_apply_all(s, x[i + off], i + off)
        end
    end
    return nothing
end

# Work items `w_lo:w_hi`, item `w` being segment `p` of row `o`.
Base.@propagate_inbounds function _pw_items!(y, x, steps, shape, inner, K, seg, per_row, w_lo, w_hi)
    for w in w_lo:w_hi
        o, p = fldmod1(w, per_row)
        row = (o - 1) * inner
        lo = row + (p - 1) * seg + 1
        hi = row + min(inner, p * seg)
        _pw_segment!(y, x, steps, shape, inner, K, lo, hi, row * (K - 1))
    end
    return nothing
end

_pw_small_length(::PwPlain, y, x) = length(x)
_pw_small_length(::PwExpand, y, x) = length(x)
_pw_small_length(::PwReduce, y, x) = length(y)

function _pw_kernel!(y, x, steps, shape, layout, threaded::Bool)
    n = _pw_small_length(shape, y, x)
    inner, K = shape isa PwPlain ? (n, 1) : layout
    seg = _pw_segment_length(steps, inner)
    per_row = cld(inner, seg)
    nitems = per_row * (n ÷ inner)
    if threaded && Threads.nthreads() > 1 && nitems > 1
        nchunks = min(Threads.nthreads(), nitems)
        per = cld(nitems, nchunks)
        quiesce_foreign_pools()
        Threads.@threads for t in 1:nchunks
            @inbounds _pw_items!(
                y, x, steps, shape, inner, K, seg, per_row, (t - 1) * per + 1, min(nitems, t * per)
            )
        end
    else
        @inbounds _pw_items!(y, x, steps, shape, inner, K, seg, per_row, 1, nitems)
    end
    return y
end

# Whether the run can be fused for these arrays. The kernels above index elements, so only
# `Array`s qualify here; a device backend adds a method for its own arrays.
_pw_fits(y, x, steps, shape, layout) = false
_pw_fits(y::Array, x::Array, steps, shape, layout) = _pw_shape_fits(y, x, steps, shape, layout)

function _pw_shape_fits(y, x, steps, shape, layout)
    layout === nothing && return false
    n = _pw_small_length(shape, y, x)
    if shape isa PwPlain
        length(y) == n || return false
        return _pw_steps_fit(steps, n)
    end
    inner, K = layout
    inner > 0 && n % inner == 0 || return false
    length(shape isa PwExpand ? y : x) == n * K || return false
    return _pw_steps_fit(steps, n * K)
end

# ─── Grouping ────────────────────────────────────────────────────────────────
#
# `_pw_run_length(ops)` is how many of the leading operators form one fused run; 1 means the
# first operator runs on its own. It is resolved by dispatch on the operator types alone.

_pw_count_maps(::Tuple{}) = 0
_pw_count_maps(ops::Tuple) = _pw_count_maps(_pw_kind(first(ops)), ops)
_pw_count_maps(::PwMapKind, ops::Tuple) = 1 + _pw_count_maps(Base.tail(ops))
_pw_count_maps(::PwKind, ops::Tuple) = 0

# Maps, then one reduction if one follows them.
function _pw_count_maps_reduce(ops::Tuple)
    m = _pw_count_maps(ops)
    return m + _pw_is_reduce(_pw_drop(ops, Val(m)))
end
_pw_is_reduce(::Tuple{}) = 0
_pw_is_reduce(ops::Tuple) = _pw_kind(first(ops)) isa PwReduceKind ? 1 : 0

_pw_run_length(ops::Tuple) = _pw_run_length(_pw_kind(first(ops)), ops)
_pw_run_length(::PwKind, ops::Tuple) = 1
_pw_run_length(::PwMapKind, ops::Tuple) = _pw_count_maps_reduce(ops)
_pw_run_length(::PwExpandKind, ops::Tuple) = 1 + _pw_count_maps(Base.tail(ops))
_pw_run_length(::PwEpilogueKind, ops::Tuple) = 1 + _pw_count_maps_reduce(Base.tail(ops))

@inline _pw_take(t::Tuple, ::Val{0}) = ()
@inline _pw_take(t::Tuple, ::Val{K}) where {K} = (first(t), _pw_take(Base.tail(t), Val(K - 1))...)
@inline _pw_drop(t::Tuple, ::Val{0}) = t
@inline _pw_drop(t::Tuple, ::Val{K}) where {K} = _pw_drop(Base.tail(t), Val(K - 1))

_pw_any_threaded(ops::Tuple) = any(is_threaded, ops)

# ─── Execution ───────────────────────────────────────────────────────────────

"""
	POINTWISE_FUSION

A `Ref{Bool}`, `true` by default: a `Compose` runs each run of consecutive pointwise operators
as one fused loop. With `AbstractOperators.POINTWISE_FUSION[] = false`, every operator of a
chain runs its own `mul!` and writes its own intermediate buffer, as operators that do not opt
in always do.

It is read each time a chain is applied, so a change takes effect at the next application,
without recompiling. Both settings compute the same result bit for bit; the switch exists to
measure what fusion is worth.
"""
const POINTWISE_FUSION = Ref(true)

# Runs `ops` one after another on `x`: the output of `ops[k]` goes to `bufs[k]`, the output of
# the last one to `y`. Leading pointwise runs are fused unless `POINTWISE_FUSION[]` is false.
@inline _pw_chain!(y, ops::Tuple{Any}, ::Tuple{}, x) = mul!(y, first(ops), x)
@inline function _pw_chain!(y, ops::Tuple, bufs::Tuple, x)
    POINTWISE_FUSION[] || return _pw_chain!(y, ops, bufs, x, Val(1))
    return _pw_chain!(y, ops, bufs, x, Val(_pw_run_length(ops)))
end

@inline function _pw_chain!(y, ops::Tuple, bufs::Tuple, x, ::Val{1})
    mul!(first(bufs), first(ops), x)
    return _pw_chain!(y, Base.tail(ops), Base.tail(bufs), first(bufs))
end

@inline function _pw_chain!(y, ops::Tuple, bufs::Tuple, x, ::Val{K}) where {K}
    run = _pw_take(ops, Val(K))
    rest = _pw_drop(ops, Val(K))
    if isempty(rest)
        _pw_run!(y, run, bufs, x)
        return y
    else
        out = bufs[K]
        _pw_run!(out, run, _pw_take(bufs, Val(K - 1)), x)
        return _pw_chain!(y, rest, _pw_drop(bufs, Val(K)), out)
    end
end

# Unfused: what `Compose` does for any run.
@inline _pw_sequential!(y, ops::Tuple{Any}, ::Tuple{}, x) = mul!(y, first(ops), x)
@inline function _pw_sequential!(y, ops::Tuple, bufs::Tuple, x)
    mul!(first(bufs), first(ops), x)
    return _pw_sequential!(y, Base.tail(ops), Base.tail(bufs), first(bufs))
end

# One fused run of at least two operators, `bufs` holding the buffers between them.
_pw_run!(y, run::Tuple, bufs::Tuple, x) = _pw_run!(_pw_kind(first(run)), y, run, bufs, x)

function _pw_run!(::PwExpandKind, y, run::Tuple, bufs::Tuple, x)
    steps = _pw_collect_steps(Base.tail(run))
    layout = _pw_layout(first(run))
    if _pw_fits(y, x, steps, PwExpand(), layout)
        return _pw_kernel!(y, x, steps, PwExpand(), layout, _pw_any_threaded(run))
    end
    return _pw_sequential!(y, run, bufs, x)
end

function _pw_run!(::PwMapKind, y, run::Tuple, bufs::Tuple, x)
    steps, shape, layout = _pw_plan_maps(run, ())
    if _pw_fits(y, x, steps, shape, layout)
        return _pw_kernel!(y, x, steps, shape, layout, _pw_any_threaded(run))
    end
    return _pw_sequential!(y, run, bufs, x)
end

# The core writes to the first buffer of the run, and the fused pass reads it from there.
function _pw_run!(::PwEpilogueKind, y, run::Tuple, bufs::Tuple, x)
    op = first(run)
    core = first(bufs)
    steps, shape, layout = _pw_plan_maps(Base.tail(run), (_pw_epilogue(op),))
    if _pw_fits(y, core, steps, shape, layout)
        _pw_core!(core, op, x)
        return _pw_kernel!(y, core, steps, shape, layout, _pw_any_threaded(run))
    end
    return _pw_sequential!(y, run, bufs, x)
end

# Steps, loop shape and layout of a run of maps that may end in a reduction, after the steps
# `lead`.
function _pw_plan_maps(maps::Tuple, lead::Tuple)
    if _pw_kind(last(maps)) isa PwReduceKind
        return (lead..., _pw_collect_steps(Base.front(maps))...), PwReduce(), _pw_layout(last(maps))
    else
        return (lead..., _pw_collect_steps(maps)...), PwPlain(), (0, 1)
    end
end

_pw_collect_steps(::Tuple{}) = ()
_pw_collect_steps(ops::Tuple) = (_pw_steps(first(ops))..., _pw_collect_steps(Base.tail(ops))...)

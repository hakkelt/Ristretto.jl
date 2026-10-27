# Host round-trip.
#
# A handful of operators cannot run on a device under any formulation available here: they
# call into a C library (OSQP), into a sparse factorization with no device equivalent
# (SPQR), into a packed-storage LAPACK routine reached by `ccall` with a `StridedVector`
# signature, or into an algorithm that is sequential in the length of the input with no
# independent blocks to spread across work-items (a single long `TotalVariation1D`).
#
# The alternative to a round-trip would be to throw, and that is the worse answer: it turns
# "this operator is not accelerated" into "this operator does not work", and it forces every
# caller who assembles a mixed problem to special-case one term. Copying to the host, running
# the existing CPU algorithm and copying back keeps the whole library usable on device data,
# at a cost that is visible and documented rather than hidden.
#
# What this deliberately does *not* do is convert the operator. Only the input and output
# are moved. An operator carrying its own data -- `LeastSquares(A, b)`, `IndAffine(A, b)` --
# is expected to have been built from host arrays, which is the realistic
# case for exactly the operators that end up here; its factorization is computed once at
# construction and would be destroyed by moving it per call.

"""
    host_prox!(y, f, x, gamma) -> value

Run `prox!` on the host and copy the result back into `y`. For operators with no device
implementation; see [`supports_device`](@ref) for which those are.

Warns once per operator type, because the difference between "runs on the GPU" and "is
copied to the CPU and back on every iteration" is a performance cliff that should not be
discovered by profiling.
"""
function host_prox!(y, f, x, gamma)
    warn_host_roundtrip(typeof(f))
    xh = host_copy(x)
    yh = similar(xh)
    v = prox!(yh, host_operator(f, xh), xh, gamma)
    copyto!(y, yh)
    return v
end

"""
    host_call(f, x)

Evaluate `f(x)` on the host. The call-operator counterpart of [`host_prox!`](@ref).
"""
host_call(f, x) = (warn_host_roundtrip(typeof(f)); xh = host_copy(x); host_operator(f, xh)(xh))

"""
    host_gradient!(g, f, x)

Run `gradient!` on the host and copy the result back into `g`.
"""
function host_gradient!(g, f, x)
    warn_host_roundtrip(typeof(f))
    xh = host_copy(x)
    gh = similar(xh)
    v = gradient!(gh, host_operator(f, xh), xh)
    copyto!(g, gh)
    return v
end

# A preallocated operator carries buffers for the input it was prepared for, and its input
# signature records that input's array type, so the host copy -- an `Array` -- never matches
# it: run on the host, a preallocated `f` would throw `DimensionMismatch` on its own
# round-trip. The host side gets an operator preallocated for the host copy instead. That
# allocates on every call, which is the price of a path that copies the input anyway.
host_operator(f, xh) = is_preallocated(f) ? preallocate(f, xh) : f

# `Array(x)` is the one conversion every array type supports, device or not.
host_copy(x::AbstractArray) = Array(x)
host_copy(xs::Tuple) = map(host_copy, xs)

const _HOST_WARNED = Set{Type}()
const _HOST_WARN_LOCK = ReentrantLock()

function warn_host_roundtrip(::Type{T}) where {T}
    lock(_HOST_WARN_LOCK) do
        T in _HOST_WARNED && return nothing
        push!(_HOST_WARNED, T)
        # the set above already makes this once-per-type; the guard is what keeps a
        # per-iteration performance cliff from being discovered by profiling instead
        @warn "$(nameof(T)) has no device implementation: its input is copied to the " *
              "host, proxed there, and copied back on every call. The result is correct " *
              "but not accelerated. See the support matrix in the GPU documentation."
        return nothing
    end
    return nothing
end

"""
    device_tier(f) -> Symbol

How `f` behaves on device storage. Every operator in the package *works* on device arrays;
this says at what cost.

* `:native` — runs as device kernels, through broadcasts, reductions and device BLAS.
* `:device_lapack` — runs natively where the backend provides the factorization it needs
  (`svd!`, `eigen!`), and takes a host round-trip where it does not. Which of the two
  happens is a property of the array package, not of this one, so it cannot be decided here.
* `:host` — always a host round-trip: no device formulation exists.

The coverage test in `test/test_gpu.jl` enumerates this, so an operator added later without
a device story has to declare a tier rather than simply failing.
"""
device_tier(::Type) = :native
device_tier(f) = device_tier(typeof(f))

"""
    supports_device(f) -> Bool

Whether `f` runs natively on device storage, i.e. its tier is not `:host`.
"""
supports_device(f) = device_tier(f) !== :host

"""
    runs_natively(op, x) -> Bool

Whether `x` can be handled without leaving the device, given that the operator's algorithm
needs `op` (typically `svd!` or `eigen!`).

The question is asked of the method table rather than of the array type, so that an operator
uses a device factorization exactly when one exists: CUDA.jl defines `svd!` for `CuMatrix`,
and `NuclearNorm` should then run on the device rather than being copied to the host for no
reason.

`hasmethod` is not the test, though, and that is the subtle part: `LinearAlgebra` defines
`svd!(::AbstractMatrix)` generically, so *every* array type has a method and `hasmethod`
answers `true` for all of them -- including backends whose only route through that generic
method is scalar indexing. What distinguishes a real device implementation is that some
other package owns it, so the owning module is what gets checked.
"""
@inline function runs_natively(op::F, x) where {F}
    is_cpu_storage(typeof(x)) && return true
    hasmethod(op, Tuple{typeof(x)}) || return false
    return parentmodule(which(op, Tuple{typeof(x)})) !== LinearAlgebra
end

"""
    _host_graph_prox!(x, y, f, c, d, gamma)

The round-trip for `IndGraph`, whose `prox!` takes and returns a *pair* of arrays and so does
not fit [`host_prox!`](@ref).
"""
function _host_graph_prox!(x, y, f, c, d, gamma)
    warn_host_roundtrip(typeof(f))
    ch, dh = host_copy(c), host_copy(d)
    xh, yh = similar(ch), similar(dh)
    v = prox!(xh, yh, f, ch, dh, gamma)
    copyto!(x, xh)
    copyto!(y, yh)
    return v
end

# A calculus rule is only as device-friendly as what it wraps: `Postcompose(NuclearNorm(..))`
# has whatever tier `NuclearNorm` has, and a `SeparableSum` containing one host-only term is
# host-only in that term. Propagating the trait means the support matrix stays correct as
# operators are combined, instead of claiming `:native` for a wrapper around a round-trip.
_combine_tiers(a::Symbol, b::Symbol) =
    (a === :host || b === :host) ? :host :
    (a === :device_lapack || b === :device_lapack) ? :device_lapack : :native

_combine_tiers(ts...) = reduce(_combine_tiers, ts; init = :native)

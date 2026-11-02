# indicator of the ball of matrices with (at most) a given rank

using LinearAlgebra
using TSVD

export IndBallRank

"""
    IndBallRank(r=1)

Return the indicator function of the set of matrices of rank at most `r`:
```math
S = \\{ X : \\mathrm{rank}(X) \\leq r \\},
```
Parameter `r` must be a positive integer.
"""
struct IndBallRank{I, B}
    r::I
    buf::B
    function IndBallRank{I, B}(r::I, buf::B) where {I, B}
        return if r <= 0
            error("parameter r must be a positive integer")
        else
            new(r, buf)
        end
    end
end

is_set_indicator(f::Type{<:IndBallRank}) = true
is_proximable(f::Type{<:IndBallRank}) = true

IndBallRank(r::I = 1; buf = nothing) where {I} = IndBallRank{I, typeof(buf)}(r, buf)
IndBallRank{I}(r::I) where {I} = IndBallRank{I, Nothing}(r, nothing)

# `tsvd` has no in-place entry point, so it keeps allocating its factors on
# every call; only the product buffer can be preallocated here.
preallocate(f::IndBallRank, x::AbstractMatrix) = IndBallRank(
    f.r; buf = (sig = input_signature(x), M = similar(x, f.r, size(x, 2)))
)

function (f::IndBallRank)(x)
    R = real(eltype(x))
    maxr = minimum(size(x))
    if maxr <= f.r
        return R(0)
    end
    # The singular values come from a full SVD, not from `tsvd`: on a matrix of rank `r` exactly,
    # which is what `prox!` returns, the Lanczos bidiagonalization of `tsvd(x, r + 1)` breaks down
    # and its last value is spurious (0.76-0.91 of the largest on 20×50 matrices, where the SVD
    # gives 1e-16), so the indicator of a projection came out `Inf`. A device matrix takes the
    # SVD of its backend where there is one and the host otherwise; only the leading `r + 1`
    # values are read, on the host.
    if !is_cpu_storage(typeof(x))
        runs_natively(svdvals!, x) || return host_call(f, x)
    end
    S = Array(svdvals(x))[1:(f.r + 1)]
    # the tolerance in the following line should be customizable
    if S[end] / S[1] <= 1.0e-7
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndBallRank, x, gamma)
    R = real(eltype(x))
    check_input(f, x)
    maxr = minimum(size(x))
    if maxr <= f.r
        y .= x
        return R(0)
    end
    b = get_buffers(f, x)
    # TODO: the order of the following matrix products should depend on the shape of x
    M = b.M
    if is_cpu_storage(typeof(x))
        U, S, V = tsvd(x, f.r)
        M .= S .* V'
        mul!(y, U, M)
    else
        runs_natively(svd!, x) || return host_prox!(y, f, x, gamma)
        F = svd!(copy(x))
        M .= view(F.S, 1:f.r) .* view(F.Vt, 1:f.r, :)
        mul!(y, view(F.U, :, 1:f.r), M)
    end
    return R(0)
end

function prox_naive(f::IndBallRank, x, gamma)
    R = real(eltype(x))
    maxr = minimum(size(x))
    if maxr <= f.r
        y = x
        return y, R(0)
    end
    F = svd(x)
    y = F.U[:, 1:f.r] * (Diagonal(F.S[1:f.r]) * F.V[:, 1:f.r]')
    return y, R(0)
end

# see `device_tier` in src/utilities/hostfallback.jl
device_tier(::Type{<:IndBallRank}) = :device_lapack

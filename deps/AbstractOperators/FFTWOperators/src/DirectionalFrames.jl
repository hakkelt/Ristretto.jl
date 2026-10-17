export ShearletOp, SurfaceletOp

# Band-limited directional frames applied in the Fourier domain. Each band k is a real, even
# window Ψₖ on the DFT grid, and the windows form a partition of unity of squares, Σₖ Ψₖ² = 1.
# The analysis operator stacks the band-filtered copies of the input along one extra trailing
# dimension,
#
#     W x = (ifft(Ψ₁ ⋅ fft x), …, ifft(Ψₖ ⋅ fft x)),     Wᴴ c = Σₖ ifft(Ψₖ ⋅ fft cₖ),
#
# so WᴴW = I (a Parseval frame) while WWᴴ is a projection, not the identity. ‖W x‖₁ is therefore
# not proximable in closed form, and these operators deliberately do not claim a diagonal AAᴴ.
#
# The windows are built from one smooth step S with S(t)² + S(1 - t)² = 1 (Meyer's auxiliary
# polynomial inside a sine), used for both the radial (scale) and the angular (direction)
# partitions. Because every window is even in ξ, a real input gives real coefficients.

# Meyer's auxiliary function: ν(t) + ν(1 - t) = 1, flat to third order at 0 and 1.
_meyer_ν(t) = (t = clamp(t, zero(t), one(t)); t^4 * (35 - 84t + 70t^2 - 20t^3))
_smooth_step(t) = sinpi(_meyer_ν(t) / 2)

# Normalized DFT frequencies in [-1, 1) (1 = Nyquist), in FFT order.
_freq_axis(n::Int) = [2 * (k < (n + 1) ÷ 2 ? k : k - n) / n for k in 0:(n - 1)]

# Radial partition over r = max |ξ_d| (dyadic square annuli): the low-pass L₀ and the band-pass
# windows W₁ … W_J, coarse to fine, with L₀² + Σ W_j² = 1. L_m(r) = S(2 - r / c_m) is 1 for
# r ≤ c_m and 0 for r ≥ 2c_m, with c_m = 2^(m - J); band j is √(L_j² - L_{j-1}²), and the finest
# band takes everything above c_{J-1} = 1/2.
function _radial_windows(r::Real, J::Int)
    L(m) = m >= J ? one(r) : _smooth_step(2 - r / 2.0^(m - J))
    low = L(0)
    return low, ntuple(j -> sqrt(max(L(j)^2 - L(j - 1)^2, zero(r))), J)
end

# Periodic angular partition: `n` windows of width Δ = period / n centred at `offset + kΔ`, each
# overlapping only its two neighbours, with Σₖ Aₖ(p)² = 1 for every p.
function _angular_window(p::Real, k::Int, n::Int, period::Real, offset::Real)
    n == 1 && return one(p)
    Δ = period / n
    u = mod(p - offset - k * Δ + period / 2, period) - period / 2
    v = u / Δ
    return _smooth_step(v + 1) * _smooth_step(1 - v)
end

# Interval partition of the real line into `n` windows centred on a uniform grid of [-1, 1];
# the outer two stay 1 beyond their centres, so the partition covers slopes outside [-1, 1]
# (where the pyramid weight of `_surfacelet_filters` has nearly vanished) as well.
function _interval_window(s::Real, k::Int, n::Int)
    n == 1 && return one(s)
    Δ = 2 / n
    v = (s - (-1 + (k + 1 / 2) * Δ)) / Δ
    rise = k == 0 ? one(s) : _smooth_step(v + 1)
    fall = k == n - 1 ? one(s) : _smooth_step(1 - v)
    return (v < 0 ? rise : fall)
end

# Shearlet pseudo-angle on [0, 4): within the horizontal cone (|ξ₂| ≤ |ξ₁|) it is 1 + ξ₂/ξ₁, in
# the vertical cone 3 - ξ₁/ξ₂. It is continuous across the cone seams (the diagonals) and even
# in ξ, and it is linear in the shear parameter within each cone, so windows equispaced in it are
# equispaced in shear.
_shear_angle(ξ1, ξ2) = abs(ξ2) <= abs(ξ1) ? (iszero(ξ1) ? zero(ξ1) : 1 + ξ2 / ξ1) : 3 - ξ1 / ξ2

function _shearlet_filters(dim_in::NTuple{2, Int}, directions::NTuple{J, Int}) where {J}
    ξ1, ξ2 = _freq_axis(dim_in[1]), _freq_axis(dim_in[2])
    Ψ = zeros(Float64, dim_in..., 1 + sum(directions))
    for j2 in 1:dim_in[2], j1 in 1:dim_in[1]
        a, b = ξ1[j1], ξ2[j2]
        low, bands = _radial_windows(max(abs(a), abs(b)), J)
        Ψ[j1, j2, 1] = low
        p = _shear_angle(a, b)
        i = 1
        for j in 1:J, k in 0:(directions[j] - 1)
            i += 1
            # Centres at p = 1 + kΔ: the first window points along ξ₁, and with a direction
            # count divisible by 4 the axes and the diagonals each get a window of their own.
            Ψ[j1, j2, i] = bands[j] * _angular_window(p, k, directions[j], 4, 1)
        end
    end
    return Ψ
end

# Pyramid weights for the three dominant axes, P_n² = |ξ_n|^q / Σ_m |ξ_m|^q: smooth away from the
# origin, Σ P_n² = 1, and P_n is negligible outside the pyramid |ξ_n| ≥ max_m |ξ_m|.
const _PYRAMID_EXPONENT = 12

function _surfacelet_filters(dim_in::NTuple{3, Int}, levels::NTuple{J, Int}) where {J}
    ξ = map(_freq_axis, dim_in)
    nb = 1 + sum(l -> 3 * 4^l, levels)
    Ψ = zeros(Float64, dim_in..., nb)
    for I in CartesianIndices(dim_in)
        f = (ξ[1][I[1]], ξ[2][I[2]], ξ[3][I[3]])
        low, bands = _radial_windows(maximum(abs, f), J)
        Ψ[I, 1] = low
        w = abs.(f) .^ _PYRAMID_EXPONENT
        sw = sum(w)
        i = 1
        for j in 1:J
            K = 2^levels[j]
            for n in 1:3
                P = iszero(sw) ? sqrt(1 / 3) : sqrt(w[n] / sw)
                a, b = n == 1 ? (2, 3) : n == 2 ? (1, 3) : (1, 2)
                sa = iszero(f[n]) ? zero(f[n]) : f[a] / f[n]
                sb = iszero(f[n]) ? zero(f[n]) : f[b] / f[n]
                for kb in 0:(K - 1), ka in 0:(K - 1)
                    i += 1
                    Ψ[I, i] = bands[j] * P * _interval_window(sa, ka, K) * _interval_window(sb, kb, K)
                end
            end
        end
    end
    return Ψ
end

"""
    ShearletOp([T=Float64,] dim_in::NTuple{2,Int}; directions=(8, 8, 16), threaded=true)
    ShearletOp(x::AbstractMatrix; directions=(8, 8, 16), threaded=true)

Discrete band-limited shearlet transform of a 2D image: a Parseval frame of real, even
Fourier-domain windows, `(dim_in..., nbands) ← dim_in`, `nbands = 1 + sum(directions)`.

The frequency plane is split into a low-pass square and `length(directions)` dyadic square
annuli (coarse to fine), and annulus `j` into `directions[j]` windows equispaced in shear within
the horizontal and the vertical cone, with the windows that straddle the cone boundaries (the
diagonals) smoothly joining the two cones -- the cone-adapted, band-limited construction of the
finite shearlet transform (Häuser and Steidl, "Fast finite shearlet transform", 2012). The default
doubles the direction count every second scale, the parabolic scaling of shearlets. Windows are
smooth (C³), so every band is a localized atom in space.

`W'W = I`, but `W*W'` is not the identity (the frame is `nbands`-times redundant), so an ℓ₁
penalty on `W*x` needs a splitting method rather than a closed-form proximal step.
"""
function ShearletOp end

# The operator type behind both constructors: one Parseval frame of Fourier windows.
struct FourierFrameOp{N, T, F, P, Pi, Q, Qi, B, S} <: LinearOperator
    dim_in::NTuple{N, Int}
    filters::F                      # (dim_in..., nbands), real
    plan::P                         # in-place forward FFT over 1:N of `buf`
    iplan::Pi                       # in-place unnormalized backward FFT over 1:N of `buf`
    splan::Q                        # the same two for `spec`
    isplan::Qi
    buf::B                          # (dim_in..., nbands), complex scratch
    spec::S                         # dim_in, complex scratch (spectrum of the input)
    num_threads::Int
    params::Tuple
end

"""
    SurfaceletOp([T=Float64,] dim_in::NTuple{3,Int}; levels=(1, 1, 1), threaded=true)
    SurfaceletOp(x::AbstractArray{T,3}; levels=(1, 1, 1), threaded=true)

Undecimated, band-limited surfacelet transform of a 3D volume: a Parseval frame of real, even
Fourier-domain windows, `(dim_in..., nbands) ← dim_in`, `nbands = 1 + sum(3 .* 4 .^ levels)`.

The frequency tiling is the surfacelet one (Lu and Do, "Multidimensional directional filter
banks and surfacelets", IEEE TIP 2007): a multiscale pyramid of `length(levels)` dyadic cubic
annuli (coarse to fine), each split into three hourglass-shaped pyramids by the dominant
frequency axis `n`, and each pyramid into `2^l × 2^l` directional subbands by the two slopes
`ξ_a/ξ_n` and `ξ_b/ξ_n` (`l = levels[j]`). Lu and Do realise this tiling with an iterated
filter bank and subsample the subbands; here every window is applied directly by FFT and
nothing is subsampled, which makes the transform shift-invariant and exactly tight at the cost
of `nbands`-fold redundancy.

`W'W = I`, but `W*W'` is not the identity, so an ℓ₁ penalty on `W*x` needs a splitting method
rather than a closed-form proximal step.
"""
function SurfaceletOp end

# Make every window even on the DFT grid, Ψ(ξ) = Ψ(-ξ mod n), keeping Σₖ Ψₖ² = 1. The windows are
# even functions already; only a Nyquist row of an even-sized axis, which is its own mirror while
# the other coordinates are not, breaks the symmetry. Even windows keep real inputs real.
function _symmetrize!(Ψ::Array{Float64})
    N = ndims(Ψ) - 1
    sz = size(Ψ)[1:N]
    Ψ0 = copy(Ψ)
    for I in CartesianIndices(sz)
        M = CartesianIndex(ntuple(d -> mod(sz[d] - (I[d] - 1), sz[d]) + 1, N))
        for k in axes(Ψ, N + 1)
            Ψ[I, k] = sqrt((Ψ0[I, k]^2 + Ψ0[M, k]^2) / 2)
        end
    end
    return Ψ
end

function _fourier_frame(::Type{T}, proto::AbstractArray, Ψhost::Array{Float64}, threaded::Bool, num_threads, params) where {T}
    N = ndims(Ψhost) - 1
    dim_in = size(Ψhost)[1:N]
    R = real(T)
    C = complex(R)
    filters = copyto!(similar(proto, R, size(Ψhost)), R.(_symmetrize!(copy(Ψhost))))
    buf = similar(proto, C, size(Ψhost))
    spec = similar(proto, C, dim_in)
    nthr = _fftw_num_threads(:c2c, num_threads, threaded, length(buf))
    return _assemble_frame(T, dim_in, filters, buf, spec, nthr, params)
end

function _assemble_frame(::Type{T}, dim_in::NTuple{N, Int}, filters, buf, spec, nthr, params) where {T, N}
    plan, iplan, splan, isplan = _with_fftw_threads(nthr) do
        plan_fft!(buf, 1:N), plan_bfft!(buf, 1:N), plan_fft!(spec), plan_bfft!(spec)
    end
    return FourierFrameOp{N, T, typeof(filters), typeof(plan), typeof(iplan), typeof(splan), typeof(isplan), typeof(buf), typeof(spec)}(
        dim_in, filters, plan, iplan, splan, isplan, buf, spec, nthr, params
    )
end

function ShearletOp(
        ::Type{T}, dim_in::NTuple{2, Int}; directions::Tuple = (8, 8, 16),
        threaded::Bool = true, num_threads = nothing, array_type = Array
    ) where {T}
    all(>(0), directions) || throw(ArgumentError("every direction count must be positive, got $directions"))
    Ψ = _shearlet_filters(dim_in, Tuple(Int.(directions)))
    return _fourier_frame(T, similar(_array_wrapper_type(array_type){T}, 0), Ψ, threaded, num_threads, (:shearlet, Tuple(Int.(directions))))
end
ShearletOp(dim_in::NTuple{2, Int}; kwargs...) = ShearletOp(Float64, dim_in; kwargs...)
ShearletOp(x::AbstractMatrix{T}; kwargs...) where {T} =
    ShearletOp(T, size(x); array_type = _array_wrapper_type(typeof(x)), kwargs...)

function SurfaceletOp(
        ::Type{T}, dim_in::NTuple{3, Int}; levels::Tuple = (1, 1, 1),
        threaded::Bool = true, num_threads = nothing, array_type = Array
    ) where {T}
    all(>=(0), levels) || throw(ArgumentError("directional levels must be non-negative, got $levels"))
    Ψ = _surfacelet_filters(dim_in, Tuple(Int.(levels)))
    return _fourier_frame(T, similar(_array_wrapper_type(array_type){T}, 0), Ψ, threaded, num_threads, (:surfacelet, Tuple(Int.(levels))))
end
SurfaceletOp(dim_in::NTuple{3, Int}; kwargs...) = SurfaceletOp(Float64, dim_in; kwargs...)
SurfaceletOp(x::AbstractArray{T, 3}; kwargs...) where {T} =
    SurfaceletOp(T, size(x); array_type = _array_wrapper_type(typeof(x)), kwargs...)

# Mappings

_nbands(L::FourierFrameOp) = size(L.filters)[end]

function mul!(y::AbstractArray, L::FourierFrameOp{N, T}, x::AbstractArray) where {N, T}
    check(y, L, x)
    n = prod(L.dim_in)
    copyto!(L.spec, x)
    L.splan * L.spec
    L.buf .= L.filters .* L.spec
    L.iplan * L.buf
    if T <: Real
        y .= real.(L.buf) ./ n
    else
        y .= L.buf ./ n
    end
    return y
end

function mul!(x::AbstractArray, A::AdjointOperator{<:FourierFrameOp{N, T}}, y::AbstractArray) where {N, T}
    L = A.A
    check(x, A, y)
    n = prod(L.dim_in)
    copyto!(L.buf, y)
    L.plan * L.buf
    L.buf .*= L.filters
    sum!(L.spec, L.buf)
    L.isplan * L.spec
    if T <: Real
        x .= real.(L.spec) ./ n
    else
        x .= L.spec ./ n
    end
    return x
end

# Properties

size(L::FourierFrameOp) = ((L.dim_in..., _nbands(L)), L.dim_in)
fun_name(L::FourierFrameOp) = L.params[1] === :surfacelet ? "𝒮𝓊" : "𝒮𝒽"
domain_type(::FourierFrameOp{N, T}) where {N, T} = T
codomain_type(::FourierFrameOp{N, T}) where {N, T} = T
domain_array_type(L::FourierFrameOp{N, T}) where {N, T} = _array_wrapper_type(typeof(L.filters)){T}
codomain_array_type(L::FourierFrameOp{N, T}) where {N, T} = _array_wrapper_type(typeof(L.filters)){T}
is_thread_safe(::FourierFrameOp) = false

is_AcA_diagonal(::FourierFrameOp) = true
diag_AcA(::FourierFrameOp) = 1.0
is_full_column_rank(::FourierFrameOp) = true
is_full_row_rank(L::FourierFrameOp) = _nbands(L) == 1
AbstractOperators.has_optimized_normalop(::FourierFrameOp) = true
get_normal_op(L::FourierFrameOp) = Eye(allocate_in_domain(L))
has_fast_opnorm(::FourierFrameOp) = true
LinearAlgebra.opnorm(L::FourierFrameOp) = one(real(domain_type(L)))

# FFTW threads the plans (c2c); the count is fixed at construction, as for DFT.
is_threaded(L::FourierFrameOp) = L.num_threads > 1
supports_threading(::FourierFrameOp) = true

function _copy_operator_impl(L::FourierFrameOp{N, T}; storage_type = nothing, threaded = nothing) where {N, T}
    new_threaded = threaded === nothing ? is_threaded(L) : threaded
    nthr = new_threaded == is_threaded(L) ? L.num_threads :
        _fftw_num_threads(:c2c, nothing, new_threaded, length(L.buf))
    if storage_type === nothing
        # The filters are read-only and shared; the scratch and the plans are per copy.
        return _assemble_frame(T, L.dim_in, L.filters, similar(L.buf), similar(L.spec), nthr, L.params)
    end
    filters = copyto!(similar(storage_type{real(T)}, size(L.filters)), L.filters)
    buf = similar(storage_type{eltype(L.buf)}, size(L.buf))
    spec = similar(storage_type{eltype(L.spec)}, size(L.spec))
    return _assemble_frame(T, L.dim_in, filters, buf, spec, nthr, L.params)
end

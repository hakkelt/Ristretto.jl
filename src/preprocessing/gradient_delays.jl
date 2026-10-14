"""
    GradientDelay

Abstract type representing trajectory gradient delay correction methods in non-Cartesian MRI.

Every method estimates the symmetric 2×2 delay tensor ``S`` of the model

```math
\\mathbf k_\\text{actual} = \\mathbf k_\\text{nominal} + S\\,\\mathbf n,
```

where ``\\mathbf n`` is the unit direction of a spoke (Peters et al. 2003, Moussavi et al. 2014,
Rosenzweig et al. 2019): the samples of a spoke really lie displaced by ``S\\mathbf n`` from where the
nominal trajectory puts them. The estimate is returned as `(dx = Sxx, dy = Syy, dxy = Sxy)` in
trajectory units (cycles/pixel, like the trajectory itself).
"""
abstract type GradientDelay end

"""
    OpposingSpokes(; fov = 1.2, max_delay = 8.0, tolerance = 5.0)

Adaptive gradient-delay estimation from opposed spokes (Block & Uecker, ISMRM 2011; quadratic
model of Moussavi et al. 2014). Each spoke is paired with the spoke closest to pointing the
opposite way. A delay moves both of them by the same amount ``\\mathbf n^T S\\mathbf n`` along their own
direction, i.e. in opposite directions in k-space, so the shift that best aligns the profile of
one spoke with the reversed profile of its partner is twice the delay. The per-spoke delays are
then fitted with ``\\mathbf n^T S \\mathbf n``, which gives the three entries of ``S``.

Opposed spokes sample the same line only if the delay moves spokes along themselves. An
anisotropic delay also moves each spoke sideways by ``\\mathbf v^T S \\mathbf n`` (``\\mathbf v`` the normal
of the spoke), in opposite directions for the two spokes of a pair, which the method neglects:
for an extended, off-centre object this biases the estimate (by ≈0.1 samples for a sideways
shift of half a sample). [`RING`](@ref) models the sideways shift.

The data must cover the full circle (at least half of the spokes need a partner within
`tolerance` degrees of the opposite direction), as golden-angle and 360° radial orderings do; a 180° ordering has no
opposed spokes — use [`RING`](@ref) for it.

Keyword arguments:
- `fov`: width of the object window the profiles are low-pass filtered with, as a fraction of
  the image field of view (the projection is cropped to it before the shift is measured);
- `max_delay`: largest delay searched, in readout samples;
- `tolerance`: largest deviation from exactly opposed directions accepted for a pair, in degrees.

The shift is found by sub-sample cross-correlation of the complex profiles, summed over coils
(and any further dimension of the k-space, e.g. slices).
"""
Base.@kwdef struct OpposingSpokes <: GradientDelay
    fov::Float64 = 1.2
    max_delay::Float64 = 8.0
    tolerance::Float64 = 5.0
end

"""
    RING(; fov = 1.2, region = 5.0, upsampling = 100, pairs = 1, init = :auto, phase = false)

Radial Intersections (RING, Rosenzweig, Holme & Uecker, MRM 81:1898–1906, 2019). With delays,
two non-parallel spokes no longer cross at ``k = 0`` but at a point where both measure the same
complex k-space value. For each spoke and its `pairs` nearest-to-orthogonal partners, the
profiles are interpolated finely around the k-space centre and the pair of positions where they
agree best (summed over coils) is taken as the intersection. Each intersection gives two linear
equations in ``(S_{xx}, S_{yy}, S_{xy})``, which are solved jointly by least squares.

Unlike [`OpposingSpokes`](@ref), RING needs no opposed spokes, so it works on any ordering
(including 180° linear radial) and from few spokes.

The per-pair intersections of the paper are only the starting point here: the tensor is refined
by Gauss–Newton on the RING condition itself, so that every pair agrees at the intersection the
common tensor predicts. That joint condition (two equations per pair, three unknowns) is
overdetermined even for a single coil, but per pair a single coil gives only two real equations
for two positions, and with an object that fills the field of view the profiles also agree at
spurious points. `init` chooses where the refinement starts:
- `:auto` (default) — `:opposing` when the spokes cover the full circle, `:grid` otherwise;
- `:intersections` — the paper's per-pair intersections, combined robustly (Tukey biweight);
  reliable for multi-coil data and compact objects, but with a single coil and an object that
  fills the field of view it can start outside the basin of the true tensor;
- `:opposing` — the [`OpposingSpokes`](@ref) estimate (full-circle orderings only); several
  times faster than the other two, since it needs no interpolated search grid;
- `:grid` — a coarse search of the joint condition over the tensor, read off the interpolated
  profiles; needs no per-pair intersection.

RING compares complex values, so it assumes all spokes share one phase. `phase = true` fits the
per-spoke phase ``\\psi_x\\cos\\theta + \\psi_y\\sin\\theta`` of gradient-induced B0 eddy currents
(Moussavi et al. 2014) alongside the tensor; without it such a phase biases the estimate,
most of all for single-coil data. The two extra unknowns make the fit more sensitive to noise,
so it is off by default.

Keyword arguments:
- `fov`: object window (fraction of the image field of view) the profiles are low-pass
  filtered with before interpolation;
- `region`: half-width of the searched window around the nominal k-space centre, in readout
  samples; it must contain the intersection points (a few samples for typical delays);
- `upsampling`: interpolation factor of the search grid (resolution `1/upsampling` samples);
- `pairs`: number of partner spokes per spoke;
- `init`: start of the joint refinement, `:auto`, `:intersections`, `:opposing` or `:grid`
  (see above);
- `phase`: also fit a per-spoke phase linear in the spoke direction.
"""
Base.@kwdef struct RING <: GradientDelay
    fov::Float64 = 1.2
    region::Float64 = 5.0
    upsampling::Int = 100
    pairs::Int = 1
    init::Symbol = :auto
    phase::Bool = false
end

"""
    correct_gradient_delays(acq::NonCartesianAcquisitionInfo; method = RING())

Estimates the gradient delay tensor ``S`` from the k-space data (see [`estimate_gradient_delays`](@ref))
and returns an updated `NonCartesianAcquisitionInfo` whose trajectory is moved to the actual
sample positions ``\\mathbf k + S\\mathbf n``.

The full displacement is applied, including the part across the spoke that an anisotropic delay
causes (as `bart traj -q … -O` does; without `-O`, BART keeps only the part along the spoke).
"""
function correct_gradient_delays(
        acq::NonCartesianAcquisitionInfo;
        method::GradientDelay = RING(),
    )
    delays = estimate_gradient_delays(acq; method)
    # A per-frame trajectory is corrected spoke by spoke, whatever frame each spoke belongs to.
    traj = acq.trajectory
    raw = unname(traj)
    corr = reshape(_apply_gradient_delays(reshape(raw, size(raw, 1), size(raw, 2), :), delays), size(raw))
    traj_corr = _rewrap_like(traj, corr)
    return NonCartesianAcquisitionInfo(acq; trajectory = traj_corr)
end

function correct_gradient_delays(acq::CartesianAcquisitionInfo; kwargs...)
    throw(ArgumentError("Gradient delay correction is only applicable to non-Cartesian acquisitions (got CartesianAcquisitionInfo)."))
end

"""
    estimate_gradient_delays(acq::NonCartesianAcquisitionInfo; method = RING())

Estimates the gradient delay tensor from the k-space data of a radial acquisition. Returns
`(dx = Sxx, dy = Syy, dxy = Sxy)` in trajectory units: a spoke with direction ``\\mathbf n`` was
actually sampled ``S\\mathbf n`` away from its nominal position.

The spokes must be straight with uniformly spaced samples; only the first two trajectory
coordinates (the in-plane ones) are used. Coils and any other k-space dimension are combined.

A k-space on a device (GPU) is not copied to the host: the readouts are projected onto the
object window there, and only the projections come to the host for the search.
"""
function estimate_gradient_delays(
        acq::NonCartesianAcquisitionInfo;
        method::GradientDelay = RING(),
    )
    # A device array stays on the device (see `_project_readouts`); the trajectory is on the host.
    kspace = acq.kspace_data isa AbstractArray ? acq.kspace_data : _adapt_any(Array, acq.kspace_data)
    traj, ksp = _pooled_spokes(acq.trajectory, kspace)
    @argcheck size(traj, 1) >= 2 "Trajectory must have at least 2 spatial dimensions (got $(size(traj, 1)))"
    return _estimate_delays_core(traj, ksp, method, acq.image_size[1])
end

# Gradient delays are a property of the readout hardware, the same in every frame, so the spokes
# of a per-frame trajectory are pooled across frames into one `(coord, sample, spoke)` trajectory
# and a matching `(sample, spoke, rest...)` k-space.
function _pooled_spokes(trajectory, kspace)
    nframe = _trajectory_frame_dims_count(trajectory, kspace)
    traj = unname(trajectory)
    ksp = unname(kspace)
    nframe == 0 && return traj, ksp
    @argcheck ndims(traj) - 1 - nframe == 2 "gradient delay estimation needs a (coord, sample, spoke) trajectory per frame"
    frames = (ndims(ksp) - nframe + 1):ndims(ksp)
    middle = 3:(ndims(ksp) - nframe)
    pooled = permutedims(ksp, (1, 2, frames..., middle...))
    return reshape(traj, size(traj, 1), size(traj, 2), :), reshape(pooled, size(ksp, 1), :, size(ksp)[middle]...)
end

"""
    _spoke_angles(traj::AbstractArray)

Azimuthal angle of each spoke, from its first-to-last sample displacement in the kx/ky plane.
"""
function _spoke_angles(traj::AbstractArray)
    Nsamples = size(traj, 2)
    Nspokes = size(traj, 3)
    kx_end = traj[1, Nsamples, 1:Nspokes] .- traj[1, 1, 1:Nspokes]
    ky_end = traj[2, Nsamples, 1:Nspokes] .- traj[2, 1, 1:Nspokes]
    return atan.(ky_end, kx_end)
end

"""
    _spoke_profiles(traj, ksp, fov, image_width) -> (angles, dr, xs, P)

Shared preprocessing of both estimators. Every spoke is described by its angle and, for each
coil, by its projection `P[:, spoke, channel]` (the centred DFT of the readout, frequencies `xs`),
cropped to the object window and phase-referenced so that

    f(u) = Σₓ P[x] exp(-2πi x u / N)

is the spoke's k-space value at the *nominal* position `u` samples from k = 0 along the spoke.
Sample spacing `dr` is in trajectory units. Referencing every spoke to its own nominal k = 0
removes any convention about which sample holds k = 0 (even or odd readouts, shifted grids).
"""
function _spoke_profiles(traj::AbstractArray, ksp::AbstractArray, fov::Real, image_width::Integer)
    N = size(traj, 2)
    Nspokes = size(traj, 3)
    @argcheck N >= 8 "gradient delay estimation needs at least 8 samples per spoke (got $N)"
    angles = _spoke_angles(traj)
    steps = [hypot(traj[1, i + 1, s] - traj[1, i, s], traj[2, i + 1, s] - traj[2, i, s]) for i in 1:(N - 1), s in 1:Nspokes]
    dr = sum(steps) / length(steps)
    @argcheck maximum(abs, steps .- dr) <= 1.0e-3 * dr "gradient delay estimation needs straight spokes with uniformly spaced samples"
    # Nominal position of sample m0 along its spoke, in samples: zero when k = 0 is sample m0.
    m0 = N ÷ 2 + 1
    c = [(traj[1, m0, s] * cos(angles[s]) + traj[2, m0, s] * sin(angles[s])) / dr for s in 1:Nspokes]
    # Projection frequencies kept: positions x / (N dr) pixels inside ±fov · image_width / 2,
    # symmetric so that x and -x are both present (the opposed-spoke match needs both).
    K = min(floor(Int, fov * image_width * N * dr / 2), (N - 1) ÷ 2)
    xs = collect(-K:K)
    P = _project_readouts(reshape(ksp, N, Nspokes, :), xs, m0)
    # f(u) = f₀(u - c): referenced to the nominal position instead of to sample m0.
    phase = [cispi(2 * x * c[s] / N) for x in xs, s in 1:Nspokes]
    @.. P = P * phase
    return angles, dr, xs, P
end

# P[x, spoke, channel] = Σₘ data[m, spoke, channel] exp(2πi x (m - m0) / N) / N, on the host in
# double precision.
function _project_readouts(data::AbstractArray, xs, m0)
    if _is_device(data)
        return _project_readouts_device(data, xs, m0)
    end
    N = size(data, 1)
    Y = similar(data, ComplexF64)
    @.. Y = data
    FFTW.bfft!(Y, 1)  # Y[q + 1] = Σₘ data[m] exp(2πi q (m - 1) / N)
    bins = [mod(x, N) + 1 for x in xs]
    shift = [cispi(-2 * x * (m0 - 1) / N) / N for x in xs]
    P = similar(Y, length(xs), size(Y, 2), size(Y, 3))
    Yx = @view Y[bins, :, :]
    @.. P = Yx * shift
    return P
end

# A matrix product rather than an FFT, since not every device array type has an FFT (JLArrays).
function _project_readouts_device(data::AbstractArray, xs, m0)
    N = size(data, 1)
    T = complex(float(real(eltype(data))))
    W = _to_storage_of(data, T[cispi(2 * x * (m - m0) / N) / N for x in xs, m in 1:N])
    P = W * reshape(data, N, :)
    return reshape(ComplexF64.(Array(P)), length(xs), size(data, 2), :)
end

function _estimate_delays_core(traj::AbstractArray, ksp::AbstractArray, method::OpposingSpokes, image_width::Integer)
    angles, dr, xs, P = _spoke_profiles(traj, ksp, method.fov, image_width)
    S = _opposing_tensor(angles, xs, P, size(traj, 2), method)
    return (dx = S[1] * dr, dy = S[2] * dr, dxy = S[3] * dr)
end

# The OpposingSpokes tensor, in samples, from the spoke profiles.
function _opposing_tensor(angles, xs, P, N, method::OpposingSpokes)
    Nspokes = length(angles)
    rev = reverse(eachindex(xs))  # xs is symmetric: xs[rev[i]] == -xs[i]
    partners = _opposed_partners(angles, deg2rad(method.tolerance))
    paired = findall(!iszero, partners)
    @argcheck length(paired) >= max(3, Nspokes ÷ 2) "OpposingSpokes found an opposed partner (within $(method.tolerance)°) for only $(length(paired)) of $Nspokes spokes; the spokes must cover the full circle — use RING() otherwise"
    step = 1 / 16
    grid = (-method.max_delay):step:(method.max_delay)
    phases = [cispi(-4 * xs[k] * d / N) for k in eachindex(xs), d in grid]
    δ = zeros(length(paired))
    @budgeted_threads threads = length(paired) > 1 for q in eachindex(paired)
        i = paired[q]
        j = partners[i]
        # Spoke i sees f(n_i (a - δ)) and its partner f(n_i (a + δ)) at the actual position a,
        # so aligning them maximises |Σₓ conj(Pᵢ[x]) Pⱼ[-x] exp(-4πi x δ / N)| over δ.
        C = zeros(ComplexF64, length(xs))
        for ch in axes(P, 3), k in eachindex(xs)
            C[k] += conj(P[k, i, ch]) * P[rev[k], j, ch]
        end
        h(d) = abs(sum(C[k] * cispi(-4 * xs[k] * d / N) for k in eachindex(xs)))
        hgrid = [abs(sum(C[k] * phases[k, g] for k in eachindex(xs))) for g in eachindex(grid)]
        δ[q] = _maximise_1d(h, grid, hgrid, step)
    end
    θ = angles[paired]
    return [cos.(θ) .^ 2 sin.(θ) .^ 2 (2 .* cos.(θ) .* sin.(θ))] \ δ
end

# For every spoke, the spoke pointing most nearly opposite to it if that is within `tol` (radians)
# of opposed, 0 otherwise.
function _opposed_partners(angles, tol)
    chord(i, s) = abs(cis(angles[s]) - cis(angles[i]))
    targets = [rem2pi(a + π, RoundNearest) for a in angles]
    nearest = _nearest_on_circle(angles, targets, 2π, 1, chord, >)
    return [abs(rem2pi(angles[j] - angles[i] - π, RoundNearest)) <= tol ? j : 0 for (i, j) in enumerate(first.(nearest))]
end

# Whether enough spokes have an opposed partner for OpposingSpokes (same threshold as there).
_full_circle(angles, tol) = count(!iszero, _opposed_partners(angles, tol)) >= max(3, length(angles) ÷ 2)

"""
    _nearest_on_circle(values, targets, period, k, score, better) -> Vector{Vector{Int}}

For every `i`, the `k` indices `s ≠ i` whose `values[s]` lie nearest to `targets[i]` on a circle
of circumference `period`, ranked by `score(i, s)` (`better(a, b)` when `a` ranks first), the
lower index first among equal scores. O(n log n): only the values around each target are scored.
"""
function _nearest_on_circle(values, targets, period, k, score, better)
    n = length(values)
    w = [mod(v, period) for v in values]
    order = sortperm(w)  # stable: equal values keep their index order
    sw = w[order]
    starts = [g for g in 1:n if g == 1 || sw[g] != sw[g - 1]]  # runs of equal values
    stops = [starts[2:end] .- 1; n]
    ngroups = length(starts)
    # Runs searched on either side of the target: `k`, plus a margin for rounding.
    reach = min(k + 2, cld(ngroups, 2))
    result = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        g0 = searchsortedfirst(view(sw, starts), mod(targets[i], period))::Int
        candidates = Int[]
        for g in unique(mod1(g0 + d, ngroups) for d in (-reach):(reach - 1))
            # k + 1 lowest indices of a run suffice (spoke i may be one of them).
            for t in starts[g]:min(stops[g], starts[g] + k)
                order[t] != i && push!(candidates, order[t])
            end
        end
        sort!(candidates; lt = (a, b) -> better(score(i, a), score(i, b)) || (score(i, a) == score(i, b) && a < b))
        result[i] = candidates[1:min(k, end)]
    end
    return result
end

# Maximum of a smooth function: the best point of `grid` (`fgrid = f.(grid)`), refined by golden
# section.
function _maximise_1d(f, grid, fgrid, step; iters = 40)
    best = grid[argmax(fgrid)]
    a, b = best - step, best + step
    g = (sqrt(5) - 1) / 2
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in 1:iters
        if f1 > f2
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = f(x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = f(x2)
        end
    end
    return (a + b) / 2
end

function _estimate_delays_core(traj::AbstractArray, ksp::AbstractArray, method::RING, image_width::Integer)
    @argcheck method.pairs >= 1 "RING needs at least one partner spoke per spoke"
    @argcheck method.upsampling >= 1 "RING upsampling must be positive"
    @argcheck method.init in (:auto, :intersections, :opposing, :grid) "RING init must be :auto, :intersections, :opposing or :grid (got :$(method.init))"
    angles, dr, xs, P = _spoke_profiles(traj, ksp, method.fov, image_width)
    init = method.init
    if init === :auto
        init = _full_circle(angles, deg2rad(OpposingSpokes().tolerance)) ? :opposing : :grid
    end
    N = size(traj, 2)
    Nspokes = length(angles)
    @argcheck Nspokes >= 2 "RING needs at least two spokes"
    pairs = _ring_pairs(angles, min(method.pairs, Nspokes - 1))
    # Nominal positions (uᵢ, uⱼ) where pair p intersects for a tensor S: Q[p] * S, linear in S;
    # Q[p] is stored row by row as a tuple.
    Q = map(pairs) do (i, j)
        ni = [cos(angles[i]), sin(angles[i])]
        nj = [cos(angles[j]), sin(angles[j])]
        Tuple(permutedims([ni -nj] \ _tensor_map(nj - ni)))
    end
    S0 = if init === :opposing
        _opposing_tensor(angles, xs, P, N, OpposingSpokes(; fov = method.fov))
    else
        # Every spoke's values on a fine grid of nominal positions around k = 0: F[:, s, ch].
        u = collect((-method.region):(1 / method.upsampling):(method.region))
        E = [cispi(-2 * x * v / N) for v in u, x in xs]
        F = reshape(E * reshape(P, length(xs), :), length(u), Nspokes, size(P, 3))
        if init === :intersections
            _ring_intersections(F, u, pairs, angles)
        else
            _ring_grid(F, u, pairs, Q, method.region, method.phase)
        end
    end
    S = _refine_ring(S0, pairs, Q, angles, xs, P, N, method.phase)
    return (dx = S[1] * dr, dy = S[2] * dr, dxy = S[3] * dr)
end

# (uᵢ, uⱼ) = Q S for one pair.
_intersection(q, S) = (q[1] * S[1] + q[2] * S[2] + q[3] * S[3], q[4] * S[1] + q[5] * S[2] + q[6] * S[3])

# S v as a linear map of (Sxx, Syy, Sxy).
_tensor_map(v) = [v[1] 0 v[2]; 0 v[2] v[1]]

# Each spoke with its `npairs` partners closest to orthogonal (mod π), as in the RING paper.
function _ring_pairs(angles, npairs)
    dist(i, s) = abs(abs(rem(angles[s] - angles[i], π)) - π / 2)
    partners = _nearest_on_circle(angles, angles .+ π / 2, π, npairs, dist, <)
    return [(i, j) for i in eachindex(angles) for j in partners[i]]
end

# The paper's estimate: for each pair, the grid positions where the two spokes agree best are
# taken as the intersection, giving two linear equations in S. With few coils two profiles can
# also agree away from the true intersection, so the equations are combined robustly.
function _ring_intersections(F, u, pairs, angles)
    power = dropdims(sum(abs2, F; dims = 3); dims = 3)
    A = zeros(2 * length(pairs), 3)
    b = zeros(2 * length(pairs))
    # Only Re Σ_ch Fᵢ[l] conj(Fⱼ[m]) is needed: a real product of the stacked real and imaginary
    # parts, half the work of the complex one.
    Fri = cat(real(F), imag(F); dims = 3)
    G = similar(Fri, length(u), length(u))
    for (p, (i, j)) in enumerate(pairs)
        # Σ_ch |Fᵢ[l] - Fⱼ[m]|² = ‖Fᵢ[l]‖² + ‖Fⱼ[m]‖² - 2 Re Σ_ch Fᵢ[l] conj(Fⱼ[m])
        @views mul!(G, Fri[:, i, :], Fri[:, j, :]')
        best, l, m = Inf, 0, 0
        for mm in eachindex(u), ll in eachindex(u)
            v = power[ll, i] + power[mm, j] - 2 * G[ll, mm]
            v < best && ((best, l, m) = (v, ll, mm))
        end
        # Intersection of the actual spokes: uᵢ nᵢ + S nᵢ = uⱼ nⱼ + S nⱼ.
        ni = (cos(angles[i]), sin(angles[i]))
        nj = (cos(angles[j]), sin(angles[j]))
        a = ni .- nj
        A[2p - 1, :] .= (a[1], 0.0, a[2])
        A[2p, :] .= (0.0, a[2], a[1])
        b[2p - 1], b[2p] = u[m] .* nj .- u[l] .* ni
    end
    return _robust_tensor_fit(A, b)
end

# Least squares on pairs of rows (one intersection each), iteratively reweighted with Tukey's
# biweight on the intersection misfit.
function _robust_tensor_fit(A, b; iters = 20)
    npair = size(A, 1) ÷ 2
    S = A \ b
    for _ in 1:iters
        r = [hypot(A[2p - 1, :]' * S - b[2p - 1], A[2p, :]' * S - b[2p]) for p in 1:npair]
        σ = 1.4826 * median(r) + eps()
        w = [abs(x) < 4.685σ ? (1 - (x / 4.685σ)^2)^2 : 0.0 for x in r]
        W = repeat(w; inner = 2)
        Snew = (A .* W) \ (b .* W)
        maximum(abs, Snew - S) < 1.0e-6 && return Snew
        S = Snew
    end
    return S
end

# Coarse search of the joint RING condition over S: every pair is compared at the intersection
# S predicts, read off the fine grid (nearest node). Each pair's mismatch is normalised by its
# energy, so that low-signal regions (far from k = 0) do not look like good matches. With
# `phase`, each pair is compared up to a common phase, which the per-spoke phase terms of the
# refinement then explain.
function _ring_grid(F, u, pairs, Q, region, phase; step = 0.25)
    r = 0.8 * region
    grid = (-r):step:r
    # Channels first: the sum over them reads contiguous memory.
    Fc = permutedims(F, (3, 1, 2))
    power = dropdims(sum(abs2, F; dims = 3); dims = 3)
    costs = zeros(length(grid), length(grid), length(grid))
    cells = CartesianIndices(costs)
    # Pairs in the outer loop keep the two spokes' profiles in cache; each point still sums its
    # pairs in order.
    blocks = collect(Iterators.partition(eachindex(costs), cld(length(costs), 8 * Threads.nthreads())))
    @budgeted_threads for block in blocks
        for p in eachindex(pairs), n in block
            c = cells[n]
            costs[n] += _ring_grid_term((grid[c[1]], grid[c[2]], grid[c[3]]), Fc, power, u, pairs[p], Q[p], phase)
        end
    end
    best = argmin(costs)
    return [grid[best[1]], grid[best[2]], grid[best[3]]]
end

# The normalised mismatch of one pair at the intersection the tensor S predicts.
function _ring_grid_term(S, Fc, power, u, (i, j), q, phase)
    h = u[2] - u[1]
    nu = length(u)
    ui, uj = _intersection(q, S)
    li = round(Int, (ui - u[1]) / h) + 1
    lj = round(Int, (uj - u[1]) / h) + 1
    # Outside the window: the largest normalised mismatch.
    (1 <= li <= nu && 1 <= lj <= nu) || return 2.0
    g = zero(eltype(Fc))
    @inbounds for ch in axes(Fc, 1)
        g += Fc[ch, li, i] * conj(Fc[ch, lj, j])
    end
    e = power[li, i] + power[lj, j]
    return e > 0 ? 1 - 2 * (phase ? abs(g) : real(g)) / e : 1.0
end

# Gauss–Newton on the RING condition itself: pair (i, j) intersects at nominal positions
# (uᵢ, uⱼ) = Q S, where both spokes must measure the same values; with `phase`, each spoke's
# values are first multiplied by exp(i (ψx cos θ + ψy sin θ)), the per-spoke phase of gradient-
# induced B0 eddy currents (Moussavi et al. 2014), fitted alongside S.
function _refine_ring(S0, pairs, Q, angles, xs, P, N, phase::Bool; iters = 30)
    nch = size(P, 3)
    ω = (-2π / N) .* xs
    Pc = permutedims(P, (1, 3, 2))  # one spoke's profiles contiguous
    # Value (or, with `slope`, derivative) of spoke s at nominal position v, every channel.
    function profile(s, v; slope = false)
        e = [cis(ω[k] * v) for k in eachindex(xs)]
        out = Vector{ComplexF64}(undef, nch)
        @inbounds for c in 1:nch
            acc = zero(ComplexF64)
            for k in eachindex(xs)
                acc += slope ? im * ω[k] * Pc[k, c, s] * e[k] : Pc[k, c, s] * e[k]
            end
            out[c] = acc
        end
        return out
    end
    np = phase ? 5 : 3
    rot(s, x) = phase ? cis(x[4] * cos(angles[s]) + x[5] * sin(angles[s])) : one(ComplexF64)
    function residual(x)
        res = Vector{ComplexF64}(undef, length(pairs) * nch)
        @budgeted_threads threads = length(pairs) > 1 for p in eachindex(pairs)
            i, j = pairs[p]
            ui, uj = _intersection(Q[p], x)
            res[((p - 1) * nch + 1):(p * nch)] .= rot(i, x) .* profile(i, ui) .- rot(j, x) .* profile(j, uj)
        end
        return res
    end
    function jacobian(x, nr)
        J = zeros(ComplexF64, nr, np)
        @budgeted_threads threads = length(pairs) > 1 for p in eachindex(pairs)
            i, j = pairs[p]
            q = Q[p]
            ui, uj = _intersection(q, x)
            rows = ((p - 1) * nch + 1):(p * nch)
            gi, gj = rot(i, x), rot(j, x)
            si, sj = gi .* profile(i, ui; slope = true), gj .* profile(j, uj; slope = true)
            for t in 1:3
                J[rows, t] .= si .* q[t] .- sj .* q[3 + t]
            end
            if phase
                fi, fj = gi .* profile(i, ui), gj .* profile(j, uj)
                J[rows, 4] .= im .* (cos(angles[i]) .* fi .- cos(angles[j]) .* fj)
                J[rows, 5] .= im .* (sin(angles[i]) .* fi .- sin(angles[j]) .* fj)
            end
        end
        return J
    end
    x = phase ? [S0; 0.0; 0.0] : collect(float.(S0))
    r = residual(x)
    cost = sum(abs2, r)
    for _ in 1:iters
        J = jacobian(x, length(r))
        Δ = -([real(J); imag(J)] \ [real(r); imag(r)])
        t = 1.0
        while t > 1.0e-3
            xnew = x + t * Δ
            rnew = residual(xnew)
            if sum(abs2, rnew) < cost
                x, r, cost = xnew, rnew, sum(abs2, rnew)
                break
            end
            t /= 2
        end
        (t <= 1.0e-3 || maximum(abs, t * Δ) < 1.0e-6) && break
    end
    return x[1:3]
end

function _apply_gradient_delays(traj::AbstractArray, delays::NamedTuple)
    traj_corr = copy(traj)
    angles = _spoke_angles(traj)
    for s in axes(traj, 3)
        nx, ny = cos(angles[s]), sin(angles[s])
        dx = delays.dx * nx + delays.dxy * ny
        dy = delays.dxy * nx + delays.dy * ny
        traj_corr[1, :, s] .+= dx
        traj_corr[2, :, s] .+= dy
    end
    return traj_corr
end

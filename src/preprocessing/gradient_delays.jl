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
- `:opposing` — the [`OpposingSpokes`](@ref) estimate (full-circle orderings only); about ten
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
    # Every frame of a per-frame trajectory is corrected alike: the spokes are corrected one by one
    # whichever frame they belong to.
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
"""
function estimate_gradient_delays(
        acq::NonCartesianAcquisitionInfo;
        method::GradientDelay = RING(),
    )
    # The estimate reads the k-space alone, and the trajectory is held on the host on every storage.
    traj, ksp = _pooled_spokes(acq.trajectory, _adapt_any(Array, acq.kspace_data))
    @argcheck size(traj, 1) >= 2 "Trajectory must have at least 2 spatial dimensions (got $(size(traj, 1)))"
    return _estimate_delays_core(traj, ksp, method, acq.image_size[1])
end

# Gradient delays are a property of the readout hardware, the same in every frame, so the spokes
# of a per-frame trajectory are pooled across frames into one `(coord, sample, spoke)` trajectory
# and a matching `(sample, spoke, rest...)` k-space. A shared trajectory is returned as it is.
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
    W = [cispi(2 * x * (m - m0) / N) / N for x in xs, m in 1:N]
    data = reshape(ComplexF64.(ksp), N, Nspokes, :)
    P = similar(data, length(xs), Nspokes, size(data, 3))
    for ch in axes(data, 3)
        @views mul!(P[:, :, ch], W, data[:, :, ch])
    end
    # f(u) = f₀(u - c): referenced to the nominal position instead of to sample m0.
    P .*= [cispi(2 * x * c[s] / N) for x in xs, s in 1:Nspokes]
    return angles, dr, xs, P
end

function _estimate_delays_core(traj::AbstractArray, ksp::AbstractArray, method::OpposingSpokes, image_width::Integer)
    angles, dr, xs, P = _spoke_profiles(traj, ksp, method.fov, image_width)
    N = size(traj, 2)
    Nspokes = length(angles)
    rev = reverse(eachindex(xs))  # xs is symmetric: xs[rev[i]] == -xs[i]
    tol = deg2rad(method.tolerance)
    δ = Float64[]
    θ = Float64[]
    for i in 1:Nspokes
        j = _opposed_partner(angles, i, tol)
        isnothing(j) && continue
        # Spoke i sees f(n_i (a - δ)) and its partner f(n_i (a + δ)) at the actual position a,
        # so aligning them maximises |Σₓ conj(Pᵢ[x]) Pⱼ[-x] exp(-4πi x δ / N)| over δ.
        C = vec(sum(conj.(P[:, i, :]) .* P[rev, j, :]; dims = 2))
        h(d) = abs(sum(C[k] * cispi(-4 * xs[k] * d / N) for k in eachindex(xs)))
        push!(δ, _maximise_1d(h, method.max_delay))
        push!(θ, angles[i])
    end
    @argcheck length(δ) >= max(3, Nspokes ÷ 2) "OpposingSpokes found an opposed partner (within $(method.tolerance)°) for only $(length(δ)) of $Nspokes spokes; the spokes must cover the full circle — use RING() otherwise"
    S = [cos.(θ) .^ 2 sin.(θ) .^ 2 (2 .* cos.(θ) .* sin.(θ))] \ δ
    return (dx = S[1] * dr, dy = S[2] * dr, dxy = S[3] * dr)
end

# The spoke pointing most nearly opposite to spoke i, if it is within `tol` (radians) of opposed.
function _opposed_partner(angles, i, tol)
    j = argmax(s -> abs(cis(angles[s]) - cis(angles[i])), eachindex(angles))
    return abs(rem2pi(angles[j] - angles[i] - π, RoundNearest)) <= tol ? j : nothing
end

# Whether enough spokes have an opposed partner for OpposingSpokes (same threshold as there).
_full_circle(angles, tol) = count(i -> !isnothing(_opposed_partner(angles, i, tol)), eachindex(angles)) >= max(3, length(angles) ÷ 2)

# Maximum of a smooth function on [-r, r]: grid search, then golden-section refinement.
function _maximise_1d(f, r::Real; step = 1 / 16, iters = 40)
    grid = (-r):step:r
    best = grid[argmax(f.(grid))]
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
    # Nominal positions (uᵢ, uⱼ) where pair p intersects for a tensor S: Q[p] * S, linear in S.
    Q = map(pairs) do (i, j)
        ni = [cos(angles[i]), sin(angles[i])]
        nj = [cos(angles[j]), sin(angles[j])]
        [ni -nj] \ _tensor_map(nj - ni)
    end
    S0 = if init === :opposing
        o = _estimate_delays_core(traj, ksp, OpposingSpokes(; fov = method.fov), image_width)
        [o.dx, o.dy, o.dxy] ./ dr
    else
        # Every spoke's values on a fine grid of nominal positions around k = 0: F[:, s, ch].
        u = collect((-method.region):(1 / method.upsampling):(method.region))
        E = [cispi(-2 * x * v / N) for v in u, x in xs]
        F = similar(P, length(u), Nspokes, size(P, 3))
        for ch in axes(P, 3)
            @views mul!(F[:, :, ch], E, P[:, :, ch])
        end
        if init === :intersections
            _ring_intersections(F, u, pairs, angles)
        else
            _ring_grid(F, u, pairs, Q, method.region, method.phase)
        end
    end
    S = _refine_ring(S0, pairs, Q, angles, xs, P, N, method.phase)
    return (dx = S[1] * dr, dy = S[2] * dr, dxy = S[3] * dr)
end

# S v as a linear map of (Sxx, Syy, Sxy).
_tensor_map(v) = [v[1] 0 v[2]; 0 v[2] v[1]]

# Each spoke with its `npairs` partners closest to orthogonal (mod π), as in the RING paper.
function _ring_pairs(angles, npairs)
    pairs = Tuple{Int, Int}[]
    for i in eachindex(angles)
        dist(s) = s == i ? Inf : abs(abs(rem(angles[s] - angles[i], π)) - π / 2)
        for j in partialsort(eachindex(angles), 1:npairs; by = dist)
            push!(pairs, (i, j))
        end
    end
    return pairs
end

# The paper's estimate: for each pair, the grid positions where the two spokes agree best are
# taken as the intersection, giving two linear equations in S. With few coils two profiles can
# also agree away from the true intersection, so the equations are combined robustly.
function _ring_intersections(F, u, pairs, angles)
    power = dropdims(sum(abs2, F; dims = 3); dims = 3)
    A = zeros(2 * length(pairs), 3)
    b = zeros(2 * length(pairs))
    G = similar(F, length(u), length(u))
    for (p, (i, j)) in enumerate(pairs)
        # Σ_ch |Fᵢ[l] - Fⱼ[m]|² = ‖Fᵢ[l]‖² + ‖Fⱼ[m]‖² - 2 Re Σ_ch Fᵢ[l] conj(Fⱼ[m])
        @views mul!(G, F[:, i, :], F[:, j, :]')
        best, l, m = Inf, 0, 0
        for mm in eachindex(u), ll in eachindex(u)
            v = power[ll, i] + power[mm, j] - 2 * real(G[ll, mm])
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
    h = u[2] - u[1]
    nu = length(u)
    power = dropdims(sum(abs2, F; dims = 3); dims = 3)
    nch = size(F, 3)
    cost(S) = begin
        c = 0.0
        for (p, (i, j)) in enumerate(pairs)
            ui, uj = Q[p] * S
            li = round(Int, (ui - u[1]) / h) + 1
            lj = round(Int, (uj - u[1]) / h) + 1
            if !(1 <= li <= nu && 1 <= lj <= nu)
                c += 2.0  # outside the window: the largest normalised mismatch
                continue
            end
            g = zero(eltype(F))
            for ch in 1:nch
                g += F[li, i, ch] * conj(F[lj, j, ch])
            end
            e = power[li, i] + power[lj, j]
            c += e > 0 ? 1 - 2 * (phase ? abs(g) : real(g)) / e : 1.0
        end
        c
    end
    r = 0.8 * region
    grid = (-r):step:r
    best, S = Inf, zeros(3)
    for sxy in grid, syy in grid, sxx in grid
        c = cost([sxx, syy, sxy])
        c < best && ((best, S) = (c, [sxx, syy, sxy]))
    end
    return S
end

# Gauss–Newton on the RING condition itself: pair (i, j) intersects at nominal positions
# (uᵢ, uⱼ) = Q S, where both spokes must measure the same values; with `phase`, each spoke's
# values are first multiplied by exp(i (ψx cos θ + ψy sin θ)), the per-spoke phase of gradient-
# induced B0 eddy currents (Moussavi et al. 2014), fitted alongside S.
function _refine_ring(S0, pairs, Q, angles, xs, P, N, phase::Bool; iters = 30)
    nch = size(P, 3)
    ω = (-2π / N) .* xs
    value(s, v) = [sum(P[k, s, c] * cis(ω[k] * v) for k in eachindex(xs)) for c in 1:nch]
    slope(s, v) = [sum(im * ω[k] * P[k, s, c] * cis(ω[k] * v) for k in eachindex(xs)) for c in 1:nch]
    np = phase ? 5 : 3
    rot(s, x) = phase ? cis(x[4] * cos(angles[s]) + x[5] * sin(angles[s])) : one(ComplexF64)
    function residual(x)
        r = ComplexF64[]
        for (p, (i, j)) in enumerate(pairs)
            ui, uj = Q[p] * x[1:3]
            append!(r, rot(i, x) .* value(i, ui) .- rot(j, x) .* value(j, uj))
        end
        return r
    end
    function jacobian(x, nr)
        J = zeros(ComplexF64, nr, np)
        for (p, (i, j)) in enumerate(pairs)
            ui, uj = Q[p] * x[1:3]
            rows = ((p - 1) * nch + 1):(p * nch)
            gi, gj = rot(i, x), rot(j, x)
            J[rows, 1:3] = gi .* slope(i, ui) * Q[p][1:1, :] - gj .* slope(j, uj) * Q[p][2:2, :]
            if phase
                fi, fj = gi .* value(i, ui), gj .* value(j, uj)
                J[rows, 4] = im .* (cos(angles[i]) .* fi .- cos(angles[j]) .* fj)
                J[rows, 5] = im .* (sin(angles[i]) .* fi .- sin(angles[j]) .* fj)
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

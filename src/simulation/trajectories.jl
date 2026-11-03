## Non-Cartesian trajectory generators.
##
## All coordinates follow the NFFT.jl / NFFTOperators convention: normalized to `[-0.5, 0.5)`
## (open at +0.5). Returned trajectories are `NamedDimsArray`s with the coordinate axis first
## (named `:coord`, one of the two names `NonCartesianAcquisitionInfo` accepts) followed by the
## sample-layout dimensions, exactly as `NonCartesianAcquisitionInfo`/`get_encoding_operator`
## expect (see `src/acquisition_data/noncartesian_acquisition_info.jl`).

"""
    _open_range(lo, hi, n) -> Vector{Float32}

`n` points strictly inside `(lo, hi)`, evenly spaced as if `range(lo, hi; length = n + 2)` with
the two endpoints dropped. Used to keep radial readout coordinates inside NFFT.jl's half-open
`[-0.5, 0.5)` domain even at the extreme sample.
"""
_open_range(lo::Real, hi::Real, n::Integer) = Float32.(range(lo, hi; length = n + 2)[2:(end - 1)])

"""
    RadialOrdering

Abstract supertype of the spoke-to-spoke angle increments [`radial_trajectory`](@ref) accepts:
[`LinearOrdering`](@ref), [`GoldenAngle`](@ref) and [`TinyGoldenAngle`](@ref).
"""
abstract type RadialOrdering end

"""
    LinearOrdering() <: RadialOrdering

Spokes spaced uniformly over `[0, π)` — sequential (linear) profile order. Every spoke count has
to be acquired in full before k-space is covered evenly.
"""
struct LinearOrdering <: RadialOrdering end

"""
    GoldenAngle() <: RadialOrdering

Successive spokes rotated by the golden angle `π · (√5 - 1)/2 ≈ 111.246°` (Winkelmann et al., "An
optimal radial profile order based on the golden ratio for time-resolved MRI", IEEE Trans. Med.
Imaging 26(1):68-76, 2007). Any prefix of the sequence covers k-space near-uniformly, so one
trajectory can be retrospectively under-sampled to any spoke count without regenerating it.
"""
struct GoldenAngle <: RadialOrdering end

"""
    TinyGoldenAngle(index::Int = 2) <: RadialOrdering

The `index`-th member of the tiny-golden-angle family, `π / (φ + index - 1)` with `φ = (1 + √5)/2`
(Wundrak et al., Magn. Reson. Med. 2015 / IEEE Trans. Med. Imaging 2015). Consecutive spokes stay
closer together than with the standard golden angle, which is what sliding-window / view-sharing
reconstructions want, while the golden ratio's incremental-coverage property is retained.

`index = 1` is the standard golden angle, i.e. `TinyGoldenAngle(1)` and [`GoldenAngle`](@ref)
produce the same spokes; the default `2` is the first genuinely *tiny* member.
"""
struct TinyGoldenAngle <: RadialOrdering
    index::Int
    function TinyGoldenAngle(index::Integer = 2)
        @argcheck index >= 1 "the tiny-golden-angle index must be >= 1"
        return new(Int(index))
    end
end

"""
    radial_trajectory(nsamples::Int, nspokes::Int;
        ordering::RadialOrdering = GoldenAngle(), extent::Real = 0.5, center_out::Bool = false)
    -> NamedDimsArray{(:coord, :sample, :spoke)}

Generate a 2D radial (projection-reconstruction) k-space trajectory: `nspokes` straight lines
through the k-space center, each sampled at `nsamples` points spanning `(-extent, extent)`
(normalized units, NFFT.jl convention).

`ordering` controls the spoke-to-spoke angle increment (spokes are lines, so angles are taken
modulo `π`): [`LinearOrdering`](@ref), [`GoldenAngle`](@ref) (the default) or
[`TinyGoldenAngle`](@ref), which carries its own family index.

`center_out = true` acquires half spokes instead (center-out radial, as in UTE): each spoke runs
from the k-space center to `extent`, its first sample on `k = 0`. A half spoke is a ray, so angles
are taken modulo `2π` and every increment above doubles — `2π/φ` for the golden angle, `2π/(φ + N
- 1)` for the tiny golden angles, `2π/nspokes` for linear ordering.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :spoke)`.
"""
function radial_trajectory(
        nsamples::Int, nspokes::Int;
        ordering::RadialOrdering = GoldenAngle(), extent::Real = 0.5, center_out::Bool = false,
    )
    @argcheck nsamples > 0 "nsamples must be positive"
    @argcheck nspokes > 0 "nspokes must be positive"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    angles = _radial_spoke_angles(nspokes, ordering, center_out ? 2π : π)
    r = _spoke_radii(nsamples, extent, center_out)
    traj = Array{Float32}(undef, 2, nsamples, nspokes)
    for (s, θ) in enumerate(angles)
        traj[1, :, s] .= r .* Float32(cos(θ))
        traj[2, :, s] .= r .* Float32(sin(θ))
    end
    return NamedDimsArray{(:coord, :sample, :spoke)}(traj)
end

# `period` is `π` for full spokes (lines) and `2π` for half spokes (rays).
_radial_spoke_angles(nspokes::Integer, ::LinearOrdering, period::Real) =
    range(0, period; length = nspokes + 1)[1:nspokes]
_radial_spoke_angles(nspokes::Integer, ::GoldenAngle, period::Real) =
    _angle_sequence(nspokes, period * (sqrt(5) - 1) / 2, period)
_radial_spoke_angles(nspokes::Integer, o::TinyGoldenAngle, period::Real) =
    _angle_sequence(nspokes, period / ((1 + sqrt(5)) / 2 + o.index - 1), period)

_angle_sequence(nspokes::Integer, step::Real, period::Real) = [mod((n - 1) * step, period) for n in 1:nspokes]

"""
    _spoke_radii(nsamples, extent, center_out) -> Vector{Float32}

Signed readout positions along a spoke: `nsamples` points strictly inside `(-extent, extent)` for a
full spoke, or `extent · (0, 1, …, nsamples - 1)/nsamples` for a half spoke, which starts on the
k-space center and stops one sample short of `extent`.
"""
_spoke_radii(nsamples::Integer, extent::Real, center_out::Bool) =
    center_out ? Float32.(extent .* (0:(nsamples - 1)) ./ nsamples) : _open_range(-extent, extent, nsamples)

"""
    stack_of_stars_trajectory(nsamples::Int, nspokes::Int, npartitions::Int; kwargs...)
    -> NamedDimsArray{(:coord, :sample, :spoke, :partition)}

3D "stack-of-stars" trajectory: the 2D radial in-plane pattern from
[`radial_trajectory`](@ref) (`nsamples`, `nspokes` and `kwargs` are forwarded to it) is repeated
identically at every one of `npartitions` Cartesian partition-encoding (`kz`) positions —
radial sampling in-plane, conventional Cartesian phase encoding along the partition direction.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :spoke, :partition)`.
"""
function stack_of_stars_trajectory(nsamples::Int, nspokes::Int, npartitions::Int; kwargs...)
    @argcheck npartitions > 0 "npartitions must be positive"
    radial = radial_trajectory(nsamples, nspokes; kwargs...)
    raw_radial = NamedDims.unname(radial)
    kz = npartitions == 1 ? Float32[0] : Float32.(range(-0.5, 0.5; length = npartitions + 1)[1:npartitions])
    traj = Array{Float32}(undef, 3, nsamples, nspokes, npartitions)
    for p in 1:npartitions
        traj[1:2, :, :, p] .= raw_radial
        traj[3, :, :, p] .= kz[p]
    end
    return NamedDimsArray{(:coord, :sample, :spoke, :partition)}(traj)
end

"""
    kooshball_trajectory(nsamples::Int, nspokes::Int; extent::Real = 0.5, center_out::Bool = false)
    -> NamedDimsArray{(:coord, :sample, :spoke)}

Full 3D radial ("kooshball") trajectory: `nspokes` spokes through the k-space center, each
sampled at `nsamples` points spanning `(-extent, extent)`, with spoke directions distributed
quasi-uniformly over the sphere using the multidimensional golden-means ordering of
Chan et al., "Temporal stability of adaptive 3D radial MRI using multidimensional golden means",
Magn. Reson. Med. 61(2):354-363, 2009 (golden means Φ₁ ≈ 0.4656, Φ₂ ≈ 0.6823): spoke `n` has
`cos θ = 1 - 2 mod(nΦ₂, 1)` and azimuth `2π mod(nΦ₁, 1)`, so any run of consecutive spokes covers
the sphere near-uniformly. [`phyllotaxis_trajectory`](@ref) orders the spokes on a spiral instead.

`center_out = true` acquires half spokes, each from the k-space center (first sample on `k = 0`)
to one sample short of `extent`, as in 3D UTE and ZTE-like radial acquisitions.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :spoke)`.
"""
function kooshball_trajectory(nsamples::Int, nspokes::Int; extent::Real = 0.5, center_out::Bool = false)
    @argcheck nsamples > 0 "nsamples must be positive"
    @argcheck nspokes > 0 "nspokes must be positive"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    Φ1, Φ2 = 0.4656, 0.6823
    directions = [_unit_vector(acos(clamp(1 - 2 * mod(n * Φ2, 1.0), -1.0, 1.0)), 2π * mod(n * Φ1, 1.0)) for n in 0:(nspokes - 1)]
    return _radial_3d(nsamples, directions, extent, center_out)
end

"""
    phyllotaxis_trajectory(nsamples::Int, nspokes::Int;
        interleaves::Int = 1, extent::Real = 0.5, center_out::Bool = false)
    -> NamedDimsArray{(:coord, :sample, :spoke)}

3D radial trajectory on a spiral phyllotaxis (Piccini et al., "Spiral phyllotaxis: the natural way
to construct a 3D radial trajectory in MRI", Magn. Reson. Med. 66(4):1049-1056, 2011): `nspokes`
spokes through the k-space center, each sampled at `nsamples` points spanning `(-extent, extent)`.
Spoke `n` of `N` has azimuth `n · 2π(1 - 1/φ)` (the golden angle, 137.5°) and polar angle
`(π/2)√(n/N)`, which fills the upper hemisphere — a full spoke covers the opposite direction as
well.

The spokes are dealt into `interleaves` interleaves, spoke `n` to interleave `mod(n, interleaves)`,
and returned interleave by interleave. With a Fibonacci number of interleaves each one is a smooth
spiral from the pole to the equator, which is how the trajectory is acquired: one interleave per
heartbeat or per segment. (The self-gating `k_z` spoke Piccini et al. prepend to every interleave is
not added.)

`center_out = true` acquires half spokes, from the k-space center (first sample on `k = 0`) to one
sample short of `extent`. Half spokes need the whole sphere, so there the polar angle follows the
spherical Fibonacci lattice, `cos θ = 1 - (2n + 1)/N`, on the same golden-angle azimuths.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :spoke)`.
"""
function phyllotaxis_trajectory(
        nsamples::Int, nspokes::Int;
        interleaves::Int = 1, extent::Real = 0.5, center_out::Bool = false,
    )
    @argcheck nsamples > 0 "nsamples must be positive"
    @argcheck nspokes > 0 "nspokes must be positive"
    @argcheck interleaves >= 1 "a spiral phyllotaxis needs at least one interleave"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    direction(n) = center_out ? _fibonacci_direction(n, nspokes) :
        _unit_vector((π / 2) * sqrt(n / nspokes), n * _GOLDEN_ANGLE)
    order = sort(0:(nspokes - 1); by = n -> (mod(n, interleaves), n))
    return _radial_3d(nsamples, direction.(order), extent, center_out)
end

const _GOLDEN_ANGLE = 2π * (1 - 2 / (1 + sqrt(5)))

_unit_vector(θ, ϕ) = (sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))

# Point `n` (from 0) of the `N`-point spherical Fibonacci lattice.
_fibonacci_direction(n, N) = _unit_vector(acos(clamp(1 - (2n + 1) / N, -1.0, 1.0)), n * _GOLDEN_ANGLE)

function _radial_3d(nsamples, directions, extent, center_out)
    r = _spoke_radii(nsamples, extent, center_out)
    traj = Array{Float32}(undef, 3, nsamples, length(directions))
    for (s, (dx, dy, dz)) in enumerate(directions)
        traj[1, :, s] .= r .* Float32(dx)
        traj[2, :, s] .= r .* Float32(dy)
        traj[3, :, s] .= r .* Float32(dz)
    end
    return NamedDimsArray{(:coord, :sample, :spoke)}(traj)
end

"""
    SpiralVariant

Abstract supertype of the radial growth laws [`spiral_trajectory`](@ref) accepts:
[`Archimedean`](@ref) and [`VariableDensity`](@ref).
"""
abstract type SpiralVariant end

"""
    Archimedean() <: SpiralVariant

Constant angular velocity, `ρ(t) = extent · t` — the classic Archimedean spiral, with uniform
radial sample density.
"""
struct Archimedean <: SpiralVariant end

"""
    VariableDensity(exponent::Real = 2.0) <: SpiralVariant

`ρ(t) = extent · t^exponent`. An `exponent` above 1 makes `dρ/dt → 0` as `t → 0`, i.e. slower
initial radial growth, which oversamples the k-space center relative to the Archimedean spiral (a
common compressed-sensing spiral design) at the cost of undersampling the periphery. `1` reduces to
[`Archimedean`](@ref); below 1 does the reverse — denser periphery, sparser center.
"""
struct VariableDensity <: SpiralVariant
    exponent::Float64
    function VariableDensity(exponent::Real = 2.0)
        @argcheck exponent > 0 "the variable-density exponent must be positive"
        return new(Float64(exponent))
    end
end

"""
    spiral_trajectory(nsamples::Int, ninterleaves::Int;
        variant::SpiralVariant = Archimedean(), nturns::Real = 8, extent::Real = 0.5)
    -> NamedDimsArray{(:coord, :sample, :interleave)}

2D spiral k-space trajectory: `ninterleaves` rotated copies of a single spiral arm, each sampled
at `nsamples` points from the k-space center out to `extent` (interleave `i` is the base arm
rotated by `2π(i-1)/ninterleaves`).

`variant` selects the radial growth law — [`Archimedean`](@ref) or [`VariableDensity`](@ref),
which carries its own exponent. `t` runs linearly over `[0, 1]` along the arm, and sample density
near radius `ρ` scales with `1/(dρ/dt)` there.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :interleave)`.
"""
function spiral_trajectory(
        nsamples::Int, ninterleaves::Int;
        variant::SpiralVariant = Archimedean(), nturns::Real = 8, extent::Real = 0.5,
    )
    @argcheck nsamples > 1 "nsamples must be > 1"
    @argcheck ninterleaves > 0 "ninterleaves must be positive"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    # `t` excludes 1: at t=1, ρ=extent and a spoke aligned with an axis would land exactly on
    # the NFFT domain boundary `±0.5`, which is excluded (`[-0.5, 0.5)`).
    t = range(0, 1; length = nsamples + 1)[1:nsamples]
    ρ = _spiral_radius(variant, extent, t)
    ϕbase = Float32.(2 * Float64(π) * nturns .* t)
    traj = Array{Float32}(undef, 2, nsamples, ninterleaves)
    for i in 1:ninterleaves
        ϕ0 = Float32(2π * (i - 1) / ninterleaves)
        traj[1, :, i] .= ρ .* cos.(ϕbase .+ ϕ0)
        traj[2, :, i] .= ρ .* sin.(ϕbase .+ ϕ0)
    end
    return NamedDimsArray{(:coord, :sample, :interleave)}(traj)
end

_spiral_radius(::Archimedean, extent::Real, t) = Float32.(extent .* t)
_spiral_radius(v::VariableDensity, extent::Real, t) = Float32.(extent .* t .^ v.exponent)

"""
    floret_trajectory(nsamples::Int, ninterleaves::Int;
        nhubs::Int = 3, nturns::Real = 4, max_elevation::Real = nhubs == 1 ? π / 2 : π / 4,
        extent::Real = 0.5)
    -> NamedDimsArray{(:coord, :sample, :interleave, :hub)}

3D FLORET trajectory (Fermat Looped, ORthogonally Encoded Trajectories; Pipe et al., "A new design
and rationale for 3D orthogonally oversampled k-space trajectories", Magn. Reson. Med. 66(5):
1303-1311, 2011): center-out arms, each a Fermat spiral (`ρ ∝ √ϕ`, uniform density within its
surface) wound on a cone around a hub axis, with `ninterleaves` arms per hub and `nhubs`
orthogonal hubs (axes `k_z`, `k_x`, `k_y`, in that order).

Arm `j` of a hub lies at elevation `e_j` above the plane normal to the hub axis, with `sin e_j`
spread evenly over `(-sin max_elevation, sin max_elevation)`, and starts at azimuth `j · 137.5°`
(the golden angle). Its radius grows as `extent · √t` while its azimuth turns `2π · nturns · t`, for
`t` sampled evenly on `[0, 1)`. Three hubs with `|e| ≤ 45°` cover the sphere with the overlap the
design calls orthogonal oversampling; a single hub needs `max_elevation = π/2`, which is the default
for `nhubs = 1`.

This reproduces the geometry of FLORET, not a gradient waveform: samples are evenly spaced in `t`,
not limited by gradient amplitude or slew rate.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :interleave, :hub)`.
"""
function floret_trajectory(
        nsamples::Int, ninterleaves::Int;
        nhubs::Int = 3, nturns::Real = 4, max_elevation::Real = nhubs == 1 ? π / 2 : π / 4,
        extent::Real = 0.5,
    )
    @argcheck nsamples > 1 "nsamples must be > 1"
    @argcheck ninterleaves > 0 "ninterleaves must be positive"
    @argcheck 1 <= nhubs <= 3 "nhubs must be 1, 2 or 3"
    @argcheck 0 < max_elevation <= π / 2 "max_elevation must be in (0, π/2]"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    t = range(0, 1; length = nsamples + 1)[1:nsamples]
    ρ = extent .* sqrt.(t)
    ϕ = 2π * nturns .* t
    traj = Array{Float32}(undef, 3, nsamples, ninterleaves, nhubs)
    for j in 0:(ninterleaves - 1)
        e = asin(sin(max_elevation) * (2 * (j + 0.5) / ninterleaves - 1))
        ψ = j * _GOLDEN_ANGLE
        abc = (ρ .* cos(e) .* cos.(ϕ .+ ψ), ρ .* cos(e) .* sin.(ϕ .+ ψ), ρ .* sin(e))
        # Hub `h` has its axis along the third coordinate of the cyclic shift by `h - 1`.
        for h in 1:nhubs, d in 1:3
            traj[d, :, j + 1, h] .= abc[mod1(d - h + 1, 3)]
        end
    end
    return NamedDimsArray{(:coord, :sample, :interleave, :hub)}(traj)
end

"""
    sparkling_trajectory(nsamples::Int, nshots::Int;
        ndims::Int = 2, cutoff::Real = 0.1, decay::Real = 2,
        max_step::Real = max(extent/64, 2extent/nsamples), max_curvature::Real = max_step/10,
        iterations::Int = 200, grid_size::Int = ndims == 2 ? 128 : 32, extent::Real = 0.5,
        threaded::Bool = true)
    -> NamedDimsArray{(:coord, :sample, :shot)}

SPARKLING trajectory (Spreading Projection Algorithm for Rapid K-space sampLING; Lazarus et al.,
"SPARKLING: variable-density k-space filling curves for accelerated T2*-weighted MRI", Magn. Reson.
Med. 81(6):3643-3661, 2019; 3D in Chaithya et al., IEEE Trans. Comput. Imaging 8:1-15, 2022):
`nshots` center-out shots of `nsamples` samples each, in `ndims` (2 or 3) dimensions, whose sample
distribution is pulled towards a target density while the samples repel each other.

The target density is the isotropic, radially decaying one of Lazarus et al.: `1` up to radius
`cutoff · extent`, then `(cutoff · extent/|k|)^decay`, restricted to the cube `|k|∞ < extent`.

Each iteration takes a gradient step on the SPARKLING energy — attraction to the target minus
repulsion between samples, both with the kernel `|x - y|` — and projects every shot back onto the
constraints: it starts on `k = 0`, moves at most `max_step` between samples (gradient amplitude),
and its step changes by at most `max_curvature` between samples (slew rate), all in the normalized
units of the trajectory. The gradient comes from a particle-mesh evaluation on a `grid_size^ndims`
grid: the two densities are deposited by cloud-in-cell and convolved with the kernel by FFT, and
the resulting potential is differentiated through each sample's cloud-in-cell weights, which makes
it the exact gradient of the energy on the mesh. An iteration costs `O(nshots · nsamples + grid_size^ndims log grid_size)`. The projection
is the Euclidean one of the paper, solved approximately by a fixed number of primal-dual steps; a
final tracking pass along each shot makes the result satisfy the constraints exactly.

With `threaded = true` the projection of the shots and the gradient at the samples are split over Julia's threads, and the mesh FFTs use FFTW's threads once the mesh is large
enough for them to pay. Each shot and each sample is computed exactly as in the serial loop, so the
result is deterministic and the same for every thread count.

Returned as a `NamedDimsArray` with dimension names `(:coord, :sample, :shot)`.
"""
function sparkling_trajectory(
        nsamples::Int, nshots::Int;
        ndims::Int = 2, cutoff::Real = 0.1, decay::Real = 2, extent::Real = 0.5,
        max_step::Real = max(extent / 64, 2extent / nsamples), max_curvature::Real = max_step / 10,
        iterations::Int = 200, grid_size::Int = ndims == 2 ? 128 : 32, threaded::Bool = true,
    )
    @argcheck nsamples > 1 "nsamples must be > 1"
    @argcheck nshots > 0 "nshots must be positive"
    @argcheck ndims in (2, 3) "ndims must be 2 or 3"
    @argcheck 0 < extent <= 0.5 "extent must be in (0, 0.5]"
    @argcheck 0 < cutoff <= 1 "cutoff must be in (0, 1]"
    @argcheck decay >= 0 "decay must be non-negative"
    @argcheck max_step * (nsamples - 1) >= extent "max_step is too small to reach extent"
    @argcheck max_curvature > 0 "max_curvature must be positive"
    @argcheck iterations >= 0 "iterations must be non-negative"
    @argcheck grid_size >= 8 "grid_size must be at least 8"
    D = ndims
    k = _sparkling_initial(nsamples, nshots, D, extent, max_step)
    mesh = _SparklingMesh(D, grid_size, extent, cutoff, decay; threaded)
    M = nsamples * nshots
    step0 = 0.5 * extent / M^(1 / D)
    pts = reshape(k, D, M)
    duals = (zeros(D, nsamples - 1, nshots), zeros(D, max(nsamples - 2, 0), nshots))
    g = similar(pts)
    for it in 1:iterations
        _sparkling_gradient!(g, mesh, pts; threaded)
        gmax = maximum(sqrt(sum(abs2, view(g, :, i))) for i in 1:M)
        gmax > 0 || break
        c = step0 * (1 - (it - 1) / iterations) / gmax
        @.. pts = pts - c * g
        _sparkling_project!(k, duals, extent, max_step, max_curvature; threaded)
    end
    _sparkling_make_feasible!(k, extent, max_step, max_curvature)
    return NamedDimsArray{(:coord, :sample, :shot)}(Float32.(k))
end

# Center-out Archimedean spirals that use most of the path length `max_step` allows — in the plane
# for 2D, on a cone of half-angle 30° around the shot's axis for 3D — started on golden-angle
# directions, which avoids the mirror symmetries of evenly spaced ones (their forces would cancel and
# freeze the shots on the symmetry axes). Shots that only reach `extent` at the slowest speed would
# keep their samples packed along a line, since the energy cannot lengthen a shot by itself.
function _sparkling_initial(nsamples::Int, nshots::Int, D::Int, extent, max_step)
    k = zeros(Float64, D, nsamples, nshots)
    cone = D == 2 ? 1.0 : sin(π / 6)
    L = 0.9 * max_step * (nsamples - 1)
    Θ = max(2L / (extent * cone), 1.0)
    θ = sqrt.(2Θ .* (L .* (0:(nsamples - 1)) ./ (nsamples - 1)) ./ (extent * cone))
    θ .*= Θ / θ[end]
    r = 0.99 * extent .* θ ./ Θ
    for s in 1:nshots
        if D == 2
            ϕ = θ .+ (s - 1) * _GOLDEN_ANGLE
            k[1, :, s] .= r .* cos.(ϕ)
            k[2, :, s] .= r .* sin.(ϕ)
        else
            a = collect(_fibonacci_direction(s - 1, nshots))
            u = normalize(cross(a, abs(a[3]) < 0.9 ? [0.0, 0.0, 1.0] : [1.0, 0.0, 0.0]))
            w = cross(a, u)
            for i in 1:nsamples
                dir = cos(π / 6) .* a .+ cone .* (cos(θ[i]) .* u .+ sin(θ[i]) .* w)
                k[:, i, s] .= r[i] .* dir
            end
        end
    end
    return k
end

# The target density on the mesh nodes and the FFT of the kernel `|x|`, on a grid padded so the
# circular convolution equals the linear one for every pair of nodes, together with the FFT plans and
# work arrays of the convolution.
struct _SparklingMesh{D, F, B}
    G::Int
    h::Float64
    target::Array{Float64, D}
    kernel_hat::Array{ComplexF64, D}
    diff::Array{Float64, D}
    diff_hat::Array{ComplexF64, D}
    potential::Array{Float64, D}
    forward::F
    backward::B
end

_SparklingMesh(D::Int, G, extent, cutoff, decay; threaded::Bool = true) =
    _SparklingMesh(Val(D), G, extent, cutoff, decay; threaded)

function _SparklingMesh(::Val{D}, G, extent, cutoff, decay; threaded::Bool = true) where {D}
    h = 1 / G
    # Cloud-in-cell touches nodes 1 to G + 2, so node offsets reach G + 1 and a circular convolution
    # of size 2G + 3 or more is linear; the smallest 7-smooth size keeps the FFTs fast.
    P = nextprod((2, 3, 5, 7), 2G + 3)
    node(i) = -0.5 + (i - 1) * h
    target = zeros(ntuple(_ -> P, D))
    for I in CartesianIndices(ntuple(_ -> G, D))
        x = ntuple(d -> node(I[d]), D)
        maximum(abs, x) < extent || continue
        r = sqrt(sum(abs2, x))
        target[I] = r <= cutoff * extent ? 1.0 : (cutoff * extent / r)^decay
    end
    target ./= sum(target)
    off(i) = (i <= cld(P, 2) ? i - 1 : i - 1 - P) * h
    K = zeros(ntuple(_ -> P, D))
    for I in CartesianIndices(K)
        K[I] = sqrt(sum(abs2, ntuple(e -> off(I[e]), D)))
    end
    kernel_hat = FFTW.rfft(K)
    diff = similar(target)
    diff_hat = similar(kernel_hat)
    # `ESTIMATE` picks the same plan in every run, so the result does not depend on timings.
    num_threads = FFTWOperators._fftw_num_threads(:r2c, nothing, threaded, diff)
    forward = FFTW.plan_rfft(diff; flags = FFTW.ESTIMATE, num_threads)
    backward = FFTW.plan_irfft(diff_hat, P; flags = FFTW.ESTIMATE, num_threads)
    return _SparklingMesh(G, h, target, kernel_hat, diff, diff_hat, similar(target), forward, backward)
end

# Cloud-in-cell corners of point `x` on the mesh: the node index and weight of each.
function _cic_corners(mesh::_SparklingMesh{D}, x) where {D}
    u = ntuple(d -> (x[d] + 0.5) / mesh.h, D)
    i0 = ntuple(d -> clamp(floor(Int, u[d]), 0, mesh.G), D)
    f = ntuple(d -> u[d] - i0[d], D)
    return (
        (CartesianIndex(ntuple(d -> i0[d] + c[d] + 1, D)), prod(ntuple(d -> c[d] == 1 ? f[d] : 1 - f[d], D)))
            for c in CartesianIndices(ntuple(_ -> 0:1, D))
    )
end

# The gradient of the discretised energy: the potential `|x| * (target - samples)` on the mesh, from
# one forward and one inverse FFT, differentiated through each sample's cloud-in-cell weights. The
# deposit stays serial: samples share mesh nodes, so splitting it would change the order of the sums.
# The derivative writes each sample's own column and is split over threads with `threaded`.
function _sparkling_gradient!(g, mesh::_SparklingMesh{D}, pts::AbstractMatrix; threaded::Bool = true) where {D}
    M = size(pts, 2)
    copyto!(mesh.diff, mesh.target)
    for i in 1:M, (I, w) in _cic_corners(mesh, view(pts, :, i))
        mesh.diff[I] -= w / M
    end
    mul!(mesh.diff_hat, mesh.forward, mesh.diff)
    @.. mesh.diff_hat = mesh.diff_hat * mesh.kernel_hat
    mul!(mesh.potential, mesh.backward, mesh.diff_hat)
    nblocks = threaded ? min(capacity(), M) : 1
    @budgeted_threads threads = nblocks > 1 for b in 1:nblocks
        _cic_gradient!(g, mesh, pts, (div((b - 1) * M, nblocks) + 1):div(b * M, nblocks))
    end
    return g
end

# The derivative of the cloud-in-cell weight of corner `c` along axis `d` is `±1/h` times the weights
# of the other axes.
function _cic_gradient!(g, mesh::_SparklingMesh{D}, pts, samples) where {D}
    ϕ = mesh.potential
    corners = CartesianIndices(ntuple(_ -> 0:1, Val(D)))
    @inbounds for i in samples
        u = ntuple(d -> (pts[d, i] + 0.5) / mesh.h, Val(D))
        i0 = ntuple(d -> clamp(floor(Int, u[d]), 0, mesh.G), Val(D))
        f = ntuple(d -> u[d] - i0[d], Val(D))
        for d in 1:D
            s = 0.0
            for c in corners
                w = c[d] == 1 ? 1.0 : -1.0
                for e in 1:D
                    e == d || (w *= c[e] == 1 ? f[e] : 1 - f[e])
                end
                s += w * ϕ[CartesianIndex(ntuple(e -> i0[e] + c[e] + 1, Val(D)))]
            end
            g[d, i] = s / mesh.h
        end
    end
    return g
end

_clip(v, r) = (s = sqrt(sum(abs2, v)); s > r ? v .* (r / s) : v)

# Euclidean projection of every shot onto the constraint set — `k₁ = 0`, `|kᵢ₊₁ - kᵢ| ≤ α`,
# `|kᵢ₊₁ - 2kᵢ + kᵢ₋₁| ≤ β`, `|k|∞ ≤ extent` — by the accelerated primal-dual method of Chambolle and
# Pock (2011, Algorithm 2; the objective `½‖k - p‖²` is 1-strongly convex), with the first and second
# differences as the linear operator (`‖[Δ; Δ²]‖² ≤ 20`). The dual variables `duals` persist from
# one call to the next: successive targets differ by one small gradient step, so the previous duals
# are a close starting point and the approximation error does not accumulate over the iterations.
# Shots are independent, so with `threaded` each thread projects a contiguous block of them with work
# arrays of its own; every shot's arithmetic is the same as in the serial loop.
function _sparkling_project!(k::Array{Float64, 3}, duals, extent, α, β; threaded::Bool = true)
    nshots = size(k, 3)
    nblocks = threaded ? min(capacity(), nshots) : 1
    @budgeted_threads threads = nblocks > 1 for b in 1:nblocks
        shots = (div((b - 1) * nshots, nblocks) + 1):div(b * nshots, nblocks)
        _sparkling_project_shots!(k, duals, shots, extent, α, β)
    end
    return k
end

function _sparkling_project_shots!(k::Array{Float64, 3}, duals, shots, extent, α, β)
    D, n, _ = size(k)
    x, x̄, xold, g = (similar(k, D, n) for _ in 1:4)
    # Columns `i + 1`, `i` and `i + 2` of `x̄` for `i` = 1, ..., n - 1 or n - 2, and the columns of `g`
    # the adjoint differences add into.
    x̄₊, x̄₀, x̄₊₊, x̄₊₋, x̄₀₋ = view(x̄, :, 2:n), view(x̄, :, 1:(n - 1)), view(x̄, :, 3:n), view(x̄, :, 2:(n - 1)), view(x̄, :, 1:(n - 2))
    g₊, g₀, g₊₊, g₊₋, g₀₋ = view(g, :, 2:n), view(g, :, 1:(n - 1)), view(g, :, 3:n), view(g, :, 2:(n - 1)), view(g, :, 1:(n - 2))
    for s in shots
        p, y1, y2 = view(k, :, :, s), view(duals[1], :, :, s), view(duals[2], :, :, s)
        copyto!(x, p)
        copyto!(x̄, p)
        τ = σ = 0.99 / sqrt(20)
        for _ in 1:50
            # Dual ascent, then the prox of the ball indicators' conjugates, which shrinks every
            # column's norm by `σ` times the ball's radius.
            @.. y1 = y1 + σ * (x̄₊ - x̄₀)
            n > 2 && (@.. y2 = y2 + σ * (x̄₊₊ - 2 * x̄₊₋ + x̄₀₋))
            _shrink_columns!(y1, σ * α)
            _shrink_columns!(y2, σ * β)
            # Primal descent along -[Δ; Δ²]ᵀy, then the prox of ½‖· - p‖² plus `k₁ = 0` and the box.
            fill!(g, 0)
            @.. g₊ = g₊ + y1
            @.. g₀ = g₀ - y1
            if n > 2
                @.. g₊₊ = g₊₊ + y2
                @.. g₊₋ = g₊₋ - 2 * y2
                @.. g₀₋ = g₀₋ + y2
            end
            copyto!(xold, x)
            @.. x = clamp((x - τ * g + τ * p) / (1 + τ), -extent, extent)
            x[:, 1] .= 0
            θ = 1 / sqrt(1 + 2τ)
            τ *= θ
            σ /= θ
            @.. x̄ = x + θ * (x - xold)
        end
        copyto!(p, x)
    end
    return k
end

function _shrink_columns!(Y, tr)
    @inbounds for j in axes(Y, 2)
        r2 = 0.0
        for d in axes(Y, 1)
            r2 += abs2(Y[d, j])
        end
        r = sqrt(r2)
        factor = r > tr ? 1 - tr / r : 0.0
        for d in axes(Y, 1)
            Y[d, j] *= factor
        end
    end
    return Y
end

# Track each shot's samples under the constraints exactly: start on `k = 0`, step length at most
# `α`, step change at most `β`. A shot that leaves `[-extent, extent)` is then scaled towards the
# center as a whole, which keeps both constraints. On a nearly feasible shot this moves samples
# only by the residual constraint violation.
function _sparkling_make_feasible!(k::Array{Float64, 3}, extent, α, β)
    D, n, nshots = size(k)
    hi = Float64(prevfloat(Float32(extent)))
    for s in 1:nshots
        x = zeros(D)
        v = zeros(D)
        k[:, 1, s] .= 0
        for i in 2:n
            a = _clip(view(k, :, i, s) .- x .- v, β)
            v = _clip(v .+ a, α)
            x = x .+ v
            k[:, i, s] .= x
        end
        shot = view(k, :, :, s)
        reach = maximum(abs, shot)
        reach > hi && (shot .*= hi / reach)
    end
    return k
end

"""
    DensityCompensation

Abstract base type for k-space density compensation methods.
"""
abstract type DensityCompensation end

"""
    PipeMenonDCF(; maxit = 20, edge_correction = true, edge_samples = 3) <: DensityCompensation

Iterative sample density compensation factor (DCF) estimation based on the algorithm of
Pipe & Menon (1999) using NFFT operators.

`edge_correction` repairs the two ends of the radial profile; see [`correct_dcf_edges`](@ref) for
what it does and when to switch it off.
"""
Base.@kwdef struct PipeMenonDCF <: DensityCompensation
    maxit::Int = 20
    edge_correction::Bool = true
    edge_samples::Int = 3
end

"""
    VoronoiDCF(; bounds = nothing, edge_correction = true, edge_samples = 3) <: DensityCompensation

Geometric sample density compensation calculating Voronoi cell areas for 2D k-space trajectories.
Points are clipped within `bounds = (xmin, xmax, ymin, ymax)` (defaulting to `(-0.5, 0.5, -0.5, 0.5)`).

`edge_correction` repairs the two ends of the radial profile; see [`correct_dcf_edges`](@ref) for
what it does and when to switch it off. It matters more here than for [`PipeMenonDCF`](@ref): the
cells of the outermost samples are unbounded, so what they are actually given is the area of the
clip against `bounds`, which has nothing to do with the sampling density.
"""
Base.@kwdef struct VoronoiDCF{B} <: DensityCompensation
    bounds::B = nothing
    edge_correction::Bool = true
    edge_samples::Int = 3
end

"""
    density_compensation(acq::AcquisitionInfo; method::DensityCompensation = PipeMenonDCF())

Compute the sample density compensation factors (DCF) for a non-Cartesian acquisition and return
a new `NonCartesianAcquisitionInfo` with the computed `dcf`. A per-frame trajectory (see
[`NonCartesianAcquisitionInfo`](@ref)) gets one set of weights per frame, each computed from that
frame's samples alone.

Throws an `ArgumentError` if called on Cartesian acquisition data, as Cartesian data does not use
a density compensation factor.
"""
function density_compensation(
        acq::CartesianAcquisitionInfo;
        method::DensityCompensation = PipeMenonDCF(),
    )
    throw(
        ArgumentError(
            "Density compensation is only defined for non-Cartesian acquisitions. Cartesian acquisitions do not use a density compensation factor.",
        )
    )
end

function density_compensation(
        acq::NonCartesianAcquisitionInfo;
        method::DensityCompensation = PipeMenonDCF(),
    )
    # The weights are computed from the trajectory, which is held on the host on every storage;
    # the k-space only says which storage they go to, and is never copied.
    ksp = acq.kspace_data
    array_type = isnothing(ksp) ? Array : Base.typename(_array_type_of(ksp)).wrapper
    nframe = _trajectory_frame_dims_count(acq.trajectory, ksp)
    dcf = if nframe == 0
        compute_dcf(acq.trajectory, acq.image_size, method; array_type)
    else
        _per_frame_dcf(acq.trajectory, acq.image_size, method, nframe; array_type)
    end
    return NonCartesianAcquisitionInfo(acq; dcf = isnothing(ksp) ? dcf : _to_storage_of(ksp, dcf))
end

# A per-frame trajectory's frames are separate acquisitions, each with its own sample density: the
# weights are computed frame by frame, never over the samples of every frame pooled together.
function _per_frame_dcf(trajectory, image_size, method, nframe::Int; array_type = Array)
    traj = unname(trajectory)
    nsample = ndims(traj) - 1 - nframe
    sample_axes = ntuple(_ -> Colon(), nsample)
    dcf = similar(traj, size(traj)[2:end])
    for I in CartesianIndices(size(traj)[(nsample + 2):end])
        dcf[sample_axes..., Tuple(I)...] = compute_dcf(traj[:, sample_axes..., Tuple(I)...], image_size, method; array_type)
    end
    return trajectory isa NamedDimsArray ? NamedDimsArray{dimnames(trajectory)[2:end]}(dcf) : dcf
end

"""
    compute_dcf(trajectory::AbstractArray, image_size::Tuple, method::DensityCompensation; array_type = Array)

Compute density compensation factor weights for the given host trajectory and Cartesian image grid
size, as a host array. [`PipeMenonDCF`](@ref) runs its iteration on an NFFT plan of `array_type`
storage (a device array type to run it on the device); [`VoronoiDCF`](@ref) is computed on the host.
"""
function compute_dcf(
        trajectory::AbstractArray{T},
        image_size::Tuple,
        method::PipeMenonDCF;
        array_type::Type = Array,
    ) where {T <: Real}
    NFFTTools = getfield(parentmodule(NFFTOp), :NFFTTools)

    traj_raw = trajectory isa NamedDimsArray ? unname(trajectory) : trajectory
    coord_dim = size(traj_raw, 1)
    ksp_shape = size(traj_raw)[2:end]
    traj_flat = reshape(traj_raw, coord_dim, :)

    plan = array_type <: Array ? NFFT.plan_nfft(NFFT.backend(), traj_flat, image_size) :
        NFFT.plan_nfft(NFFT.backend(), array_type, Matrix(traj_flat), image_size)
    raw_dcf = NFFTTools.sdc(plan; iters = method.maxit)
    dcf_arr = reshape(_to_host(raw_dcf), ksp_shape)
    if method.edge_correction
        dcf_arr = correct_dcf_edges(dcf_arr; edge_samples = method.edge_samples)
    end

    if trajectory isa NamedDimsArray
        return NamedDimsArray{dimnames(trajectory)[2:end]}(dcf_arr)
    end
    return dcf_arr
end

function compute_dcf(
        trajectory::AbstractArray{T},
        image_size::Tuple,
        method::VoronoiDCF;
        array_type::Type = Array,
    ) where {T <: Real}
    traj_raw = trajectory isa NamedDimsArray ? unname(trajectory) : trajectory
    coord_dim = size(traj_raw, 1)
    @argcheck coord_dim == 2 "VoronoiDCF currently supports 2D trajectories (coord_dim == 2)"

    ksp_shape = size(traj_raw)[2:end]
    traj_flat = reshape(traj_raw, 2, :)
    bounds = method.bounds !== nothing ? method.bounds : (-0.5, 0.5, -0.5, 0.5)

    raw_dcf = _compute_voronoi_2d(traj_flat; bounds)
    dcf_arr = reshape(raw_dcf, ksp_shape)
    if method.edge_correction
        dcf_arr = correct_dcf_edges(dcf_arr; edge_samples = method.edge_samples)
    end

    if trajectory isa NamedDimsArray
        return NamedDimsArray{dimnames(trajectory)[2:end]}(dcf_arr)
    end
    return dcf_arr
end

"""
    correct_dcf_edges(dcf::AbstractArray; edge_samples = 3, fit_samples = 8)

Replace the density compensation factors of the samples at the two ends of every readout by the
trend of the samples just inside them, and return the corrected weights. The first dimension of
`dcf` is the readout; every remaining dimension is treated as a separate readout.

The ends of a readout are where a density estimate stops being a density estimate, and both
estimators Ristretto ships show it:

- A sample at the end of a readout has no neighbour beyond it. [`VoronoiDCF`](@ref)'s cell there is
  unbounded, so what it is actually given is the area of the clip against `bounds` — on a radial
  trajectory the outermost sample of each spoke comes out about 50% too heavy. [`PipeMenonDCF`](@ref)'s
  iteration sees the same one-sided neighbourhood and rings over the last few samples. Left alone
  those weights amplify the noisiest, highest-frequency samples of the acquisition.
- Where a readout starts or turns at the centre of k-space, the samples of every readout pile up on
  nearly the same point, and how an estimator treats near-coincident samples decides the DC weight —
  a spike or a hole there is a scaling error on the brightest part of the image.

The correction fits `fit_samples` weights just inside each end against sample index by least squares
and evaluates the fit at the `edge_samples` positions being replaced, so a ramp stays a ramp instead
of jumping. Fitting along the readout rather than against k-space radius is what keeps it honest on
a trajectory whose density is not a function of radius alone: a golden-angle radial acquisition has
a different angular gap either side of every spoke, so its true weights differ from spoke to spoke
at the same radius, and a fit pooled over spokes would replace each end with the average of all of
them.

Switch it off with `edge_correction = false` on [`PipeMenonDCF`](@ref) or [`VoronoiDCF`](@ref) when
the weights at the ends are meant to be discontinuous, or to see what the estimator produced on its
own. The weights are also left untouched rather than guessed at when a readout is too short to hold
both bands.
"""
function correct_dcf_edges(dcf::AbstractArray{T}; edge_samples::Int = 3, fit_samples::Int = 8) where {T <: Real}
    @argcheck edge_samples >= 1 "`edge_samples` must be at least 1"
    @argcheck fit_samples >= 2 "`fit_samples` must be at least 2 for a line to be determined"
    nsamples = size(dcf, 1)
    nsamples >= 2 * (edge_samples + fit_samples) || return dcf

    corrected = collect(dcf)
    flat = reshape(corrected, nsamples, :)
    for readout in axes(flat, 2)
        w = @view flat[:, readout]
        _extrapolate_end!(w, 1:edge_samples, (edge_samples + 1):(edge_samples + fit_samples))
        _extrapolate_end!(
            w, (nsamples - edge_samples + 1):nsamples,
            (nsamples - edge_samples - fit_samples + 1):(nsamples - edge_samples),
        )
    end
    return corrected
end

# Fit `w ≈ a + b·i` over the reference samples of one readout by least squares and write it into the
# target ones. Weights are densities, so a fit that extrapolates below zero is clamped there.
function _extrapolate_end!(w::AbstractVector, target, reference)
    x = collect(float(first(reference)):float(last(reference)))
    y = float.(@view w[reference])
    x̄ = mean(x)
    ȳ = mean(y)
    b = sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
    a = ȳ - b * x̄
    for i in target
        w[i] = max(zero(eltype(w)), oftype(w[i], a + b * i))
    end
    return w
end

# 2D Voronoi polygon clipping helpers

function _clip_polygon_2d(poly::Vector{Tuple{Float64, Float64}}, n::Tuple{Float64, Float64}, c::Float64)
    out = Tuple{Float64, Float64}[]
    isempty(poly) && return out
    len = length(poly)
    for i in 1:len
        p1 = poly[i]
        p2 = poly[i == len ? 1 : i + 1]
        d1 = n[1] * p1[1] + n[2] * p1[2] - c
        d2 = n[1] * p2[1] + n[2] * p2[2] - c
        if d1 <= 1.0e-12
            push!(out, p1)
        end
        if (d1 < -1.0e-12 && d2 > 1.0e-12) || (d1 > 1.0e-12 && d2 < -1.0e-12)
            t = d1 / (d1 - d2)
            push!(out, (p1[1] + t * (p2[1] - p1[1]), p1[2] + t * (p2[2] - p1[2])))
        end
    end
    return out
end

function _polygon_area_2d(poly::Vector{Tuple{Float64, Float64}})
    len = length(poly)
    len < 3 && return 0.0
    area = 0.0
    for i in 1:len
        p1 = poly[i]
        p2 = poly[i == len ? 1 : i + 1]
        area += p1[1] * p2[2] - p2[1] * p1[2]
    end
    return abs(area) / 2.0
end

function _compute_voronoi_2d(pts::AbstractMatrix{T}; bounds = (-0.5, 0.5, -0.5, 0.5)) where {T <: Real}
    M = size(pts, 2)
    areas = zeros(T, M)
    xmin, xmax, ymin, ymax = Float64.(bounds)

    coords = [(Float64(pts[1, i]), Float64(pts[2, i])) for i in 1:M]

    Threads.@threads for i in 1:M
        p_i = coords[i]
        poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        dists = [(hypot(coords[j][1] - p_i[1], coords[j][2] - p_i[2]), j) for j in 1:M if j != i]
        sort!(dists; by = first)

        for (d, j) in dists
            isempty(poly) && break
            max_r = maximum(hypot(v[1] - p_i[1], v[2] - p_i[2]) for v in poly)
            d > 2.0 * max_r && break

            p_j = coords[j]
            n = (p_j[1] - p_i[1], p_j[2] - p_i[2])
            c = (p_j[1]^2 + p_j[2]^2 - (p_i[1]^2 + p_i[2]^2)) / 2.0
            poly = _clip_polygon_2d(poly, n, c)
        end
        areas[i] = T(_polygon_area_2d(poly))
    end
    return areas
end

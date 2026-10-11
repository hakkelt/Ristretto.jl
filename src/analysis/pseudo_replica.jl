"""
    pseudo_replica(
        acq::AcquisitionInfo,
        method::ReconstructionMethod = DirectReconstruction();
        replicas::Int = 64,
        noise_std::Real = 1.0,
        rng::AbstractRNG = Random.default_rng(),
        kwargs...,
    )

Computes Monte Carlo pseudo-replica noise propagation and SNR / g-factor maps (Robson et al. 2008, MRM 60:895-907).
Reconstructs `replicas` noisy realizations of the k-space data and returns `(mean, std, g_factor)`.

# Notes
- `scaling` must be fixed (`FixedScaling()` or `NoScaling()`) to ensure noise variance is preserved across replicas.
"""
function pseudo_replica(
        acq::AcquisitionInfo,
        method::ReconstructionMethod = DirectReconstruction();
        replicas::Int = 64,
        noise_std::Real = 1.0,
        rng::AbstractRNG = Random.default_rng(),
        kwargs...,
    )
    _reject_partitioned(acq.kspace_data, "the pseudo-replica analysis")
    @argcheck replicas >= 2 "Number of pseudo-replicas must be at least 2"

    config_kwargs = Dict{Symbol, Any}(kwargs)
    if !haskey(config_kwargs, :scaling)
        config_kwargs[:scaling] = NoScaling()
    end
    # Replicas are quiet by default -- a per-replica log repeated 64 times is noise. A
    # `ProgressBar` belongs to the replica loop rather than to each individual reconstruction, so
    # it is lifted out here; an explicitly requested `Verbose` is left on each reconstruction,
    # which is what asking for the log means.
    outer_verbosity = as_verbosity(get(config_kwargs, :verbosity, Silent()))
    config_kwargs[:verbosity] = outer_verbosity isa Verbose ? outer_verbosity : Silent()
    norm_mode = config_kwargs[:scaling]
    @argcheck norm_mode isa Union{FixedScaling, NoScaling} "pseudo_replica requires FixedScaling or NoScaling to preserve noise variance across replicas (got $(typeof(norm_mode)))"

    spatial_size = get_image_size(acq)
    N_spatial = if acq isa CartesianAcquisitionInfo
        acq.is3D ? prod(spatial_size[1:3]) : prod(spatial_size[1:2])
    else
        prod(spatial_size)
    end
    ref_std = (noise_std / sqrt(N_spatial))

    # Acceleration factor (if subsampling present). Only Cartesian acquisitions carry a
    # `subsampling` mask; non-Cartesian acceleration is folded into the trajectory, so fall
    # back to R = 1 and let the empirical std carry the g-factor there.
    R_acc = if acq isa CartesianAcquisitionInfo && !isnothing(acq.subsampling)
        mask = to_displayable_mask(acq.subsampling, (spatial_size[1], spatial_size[2]))
        Float64(length(mask) / count(mask))
    else
        1.0
    end

    raw_ksp = unname(acq.kspace_data)
    T = eltype(raw_ksp)
    ksp_size = size(raw_ksp)

    results = Array{Any}(undef, replicas)
    with_progress(outer_verbosity, replicas; desc = "Replicas ") do verbosity
        tick = progress_tick(verbosity)
        for i in 1:replicas
            # Generate complex Gaussian noise with standard deviation noise_std
            noise = (randn(rng, real(T), ksp_size) .+ im .* randn(rng, real(T), ksp_size)) .* (real(T)(noise_std / sqrt(2)))
            replica_ksp = raw_ksp .+ noise
            if acq.kspace_data isa NamedDimsArray
                replica_ksp = NamedDimsArray{dimnames(acq.kspace_data)}(replica_ksp)
            end
            replica_acq = AcquisitionInfo(acq; kspace_data = replica_ksp)
            results[i] = reconstruct(replica_acq, method; config_kwargs...)
            isnothing(tick) || tick()
        end
    end

    images = [parent(r) for r in results]
    mean_img = sum(unname, images) ./ replicas
    diff_sq_sum = sum(abs2.(unname(r) .- mean_img) for r in images)
    std_img = sqrt.(diff_sq_sum ./ (replicas - 1))

    # Compute g-factor map: g(r) = σ_recon(r) / (σ_ref * √R)
    g_factor = std_img ./ (ref_std * sqrt(R_acc))

    first_img = first(images)
    first_img isa NamedDimsArray || return (mean = mean_img, std = std_img, g_factor = g_factor)
    out_dims = dimnames(first_img)
    return (
        mean = NamedDimsArray{out_dims}(mean_img),
        std = NamedDimsArray{out_dims}(std_img),
        g_factor = NamedDimsArray{out_dims}(g_factor),
    )
end

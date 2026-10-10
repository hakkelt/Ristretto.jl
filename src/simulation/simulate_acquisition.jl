"""
    simulate_acquisition(phantom, acq_info::CartesianAcquisitionInfo;
                         inverse_crime_check = true, keep_sensitivity_maps = nothing)
    simulate_acquisition(phantom, acq_info::NonCartesianAcquisitionInfo;
                         inverse_crime_check = true, keep_sensitivity_maps = nothing)

Simulate the k-space an acquisition described by `acq_info` would measure from `phantom`.

`acq_info.image_size` is the grid the data will be reconstructed on; the phantom's spatial size
is the grid it is simulated on, and must be at least as large along every spatial axis. Both
grids cover the same field of view. When the phantom is finer, its full k-space is computed on
its own grid and only the frequencies the reconstruction grid can represent are kept (Cartesian),
or the trajectory is sampled at the same physical frequencies on the phantom grid
(non-Cartesian), scaled and phase-shifted to the reconstruction grid's voxel size and origin.
The non-Cartesian simulation then uses a more accurate NFFT than a reconstruction does, so the
data do not share its gridding error.

Simulating with the operator that later reconstructs the data is the *inverse crime*
[Kaipio & Somersalo]: the data are exactly consistent with the model, so reconstructions look
better than they would on measured data. With `inverse_crime_check = true` a phantom of the
reconstruction's size gets a warning, and so does an integer multiple of it, which leaves a
point-sampled phantom partly consistent with the model; each is checked per spatial axis, so a
phantom that is finer in-plane only is still reported for its other axes. Rasterize the phantom area-sampled
(GeometricMedicalPhantoms' `supersample` keyword) on a grid about 1.6 times finer per axis, e.g.
202² for a 128² reconstruction: in a study of the Shepp–Logan phantom its simulation error is
then at the noise level of 30 dB SNR data [Guerquin-Kern et al.]. Score reconstructions against
the same object rasterized area-sampled at `image_size`, not against a point-sampled one. Pass
`inverse_crime_check = false` where the consistency is what a test checks.

# Arguments
- `phantom`: The object, a standard array or a `NamedDimsArray`, with its spatial axes first and
  any batch axes (frames, slices, ...) after them.
- `acq_info`: The acquisition: sampling pattern or trajectory, reconstruction grid
  (`image_size`) and, optionally, coil sensitivity maps. Maps act on the phantom, so their spatial
  size must be the phantom's.

# Keywords
- `inverse_crime_check::Bool = true`: warn when the phantom grid equals the reconstruction grid,
  or is an integer multiple of it, along any spatial axis.
- `keep_sensitivity_maps::Union{Bool, Nothing} = nothing`: `true` returns the sensitivity maps,
  resampled to the reconstruction grid. Otherwise the returned acquisition carries none, since
  the maps that simulated the data are themselves a part of the inverse crime; estimate them from
  the data (`estimate_sensitivities`) instead. Without maps, a reconstruction of unnamed data
  treats the coil axis as a batch axis, so the default `nothing` logs an info message when maps
  are dropped; `false` drops them silently.

# Returns
- A copy of `acq_info` with the simulated k-space in `kspace_data`.

# References
- Kaipio, J., & Somersalo, E. (2007). *Statistical inverse problems: discretization, model
  reduction and inverse crimes.* J Comput Appl Math, 198(2), 493-504.
  https://doi.org/10.1016/j.cam.2005.09.027
- Guerquin-Kern, M., Lejeune, L., Pruessmann, K. P., & Unser, M. (2012). *Realistic analytical
  phantoms for parallel magnetic resonance imaging.* IEEE Trans Med Imaging, 31(3), 626-636.
  https://doi.org/10.1109/TMI.2011.2174158
"""
function simulate_acquisition(
        phantom, acq_info::CartesianAcquisitionInfo;
        inverse_crime_check::Bool = true, keep_sensitivity_maps::Union{Bool, Nothing} = nothing,
    )
    acq_info = _with_positional_shifts(acq_info)
    grid = _simulation_grid(phantom, acq_info)
    _check_simulation_grid(grid, acq_info.image_size, inverse_crime_check)
    _check_simulation_maps(acq_info, grid)
    ksp = if _has_unequal_sample_counts(nothing, acq_info.image_size, acq_info.subsampling)
        _simulate_partitioned_acquisition(phantom, acq_info)
    elseif grid == acq_info.image_size
        _simulate_on_grid(phantom, acq_info)
    else
        _simulate_on_finer_grid(phantom, acq_info, grid)
    end
    smaps = _returned_maps(acq_info, grid, keep_sensitivity_maps)
    return CartesianAcquisitionInfo(acq_info; kspace_data = ksp, sensitivity_maps = smaps)
end

function simulate_acquisition(
        phantom, acq_info::NonCartesianAcquisitionInfo;
        inverse_crime_check::Bool = true, keep_sensitivity_maps::Union{Bool, Nothing} = nothing,
    )
    grid = _simulation_grid(phantom, acq_info)
    _check_simulation_grid(grid, acq_info.image_size, inverse_crime_check)
    _check_simulation_maps(acq_info, grid)
    ksp = if grid == acq_info.image_size
        _simulate_on_grid(phantom, acq_info)
    else
        _simulate_on_finer_grid(phantom, acq_info, grid)
    end
    smaps = _returned_maps(acq_info, grid, keep_sensitivity_maps)
    return NonCartesianAcquisitionInfo(acq_info; kspace_data = ksp, sensitivity_maps = smaps)
end

# The k-space of `image` on the reconstruction grid itself, `size(image)[1:nspatial] == image_size`.
function _simulate_on_grid(image, acq_info::CartesianAcquisitionInfo)
    smaps = acq_info.sensitivity_maps
    if !acq_info.is3D && !isnothing(smaps) && ndims(smaps) == 4
        @argcheck ndims(image) >= 3 && size(image, 3) == size(smaps, 4) "a 2D multislice image must have the sensitivity maps' slice count along its third axis"
    end
    ksp = _kspace_template(image, acq_info)
    acq_info = CartesianAcquisitionInfo(acq_info; kspace_data = ksp)
    E = get_encoding_operator(acq_info; fast_planning = true)
    if eltype(image) <: Real
        image = complex.(image)
    end
    mul!(ksp, E, image)
    return ksp
end

# The uninitialised k-space array the acquisition produces from `image`, named when `image` is.
function _kspace_template(image, acq_info::CartesianAcquisitionInfo)
    ksp_size = get_kspace_size(image, acq_info)
    ksp = similar(image, complex(eltype(image)), ksp_size)
    if image isa NamedDimsArray
        if acq_info.is3D && isnothing(acq_info.sensitivity_maps)
            full_ksp_dims = (:kx, :ky, :kz, dimnames(image)[4:end]...)
        elseif acq_info.is3D
            full_ksp_dims = (:kx, :ky, :kz, :coil, dimnames(image)[4:end]...)
        elseif isnothing(acq_info.sensitivity_maps)
            full_ksp_dims = (:kx, :ky, dimnames(image)[3:end]...)
        else
            full_ksp_dims = (:kx, :ky, :coil, dimnames(image)[3:end]...)
        end
        if isnothing(acq_info.subsampling)
            ksp_dims = full_ksp_dims
        else
            ksp_dims = _get_dimnames_from_subsampling(
                full_ksp_dims,
                acq_info.image_size,
                acq_info.subsampling,
            )
        end
        ksp = NamedDimsArray{ksp_dims}(NamedDims.unname(ksp))
    end
    return ksp
end

# The k-space of `image` on the reconstruction grid itself, `size(image)[1:nspatial] == image_size`.
# `operator_kwargs` go to the encoding operator (the NFFT operating point).
function _simulate_on_grid(image, acq_info::NonCartesianAcquisitionInfo; operator_kwargs...)

    # The simulated k-space is `(samples..., coil, batch...)`, the batch axes being the image's
    # axes after the spatial ones. A per-frame trajectory's trailing frame axes are the last of
    # those batch axes.
    traj = acq_info.trajectory
    nspatial = acq_info.is3D ? 3 : 2
    nsample = ndims(traj) - 1 - _simulation_frame_dims_count(traj, image, nspatial, acq_info.kspace_data)
    sample_dims = size(traj)[2:(nsample + 1)]
    batch_dims = size(image)[(nspatial + 1):end]
    ncoil = isnothing(acq_info.sensitivity_maps) ? () : (size(acq_info.sensitivity_maps, nspatial + 1),)
    ksp_size = (sample_dims..., ncoil..., batch_dims...)
    ksp = similar(image, Complex{eltype(traj)}, ksp_size)
    if image isa NamedDimsArray && traj isa NamedDimsArray
        sample_dimnames = dimnames(traj)[2:(nsample + 1)]
        coil_dimnames = isnothing(acq_info.sensitivity_maps) ? () : (:coil,)
        batch_dimnames = dimnames(image)[(nspatial + 1):end]
        ksp = NamedDimsArray{(sample_dimnames..., coil_dimnames..., batch_dimnames...)}(NamedDims.unname(ksp))
    end

    acq_info = NonCartesianAcquisitionInfo(acq_info; kspace_data = ksp)
    E = get_encoding_operator(acq_info; fast_planning = true, operator_kwargs...)
    if eltype(image) <: Real
        image = complex.(image)
    end
    mul!(ksp, E, image)
    return ksp
end

# The frame axes of a trajectory being simulated. An acquisition that carries k-space has had them
# read by its constructor already. Otherwise they are the trajectory's trailing axes named like the
# image's trailing batch axes, any sample axis remaining; a plain-array trajectory has no names to
# tell a frame axis from a sample axis of the same size (24 spokes for a 24-frame cine), so it is
# shared, as the constructor reads it when the sizes alone allow both.
function _simulation_frame_dims_count(traj, image, nspatial::Int, ksp)
    isnothing(ksp) || return _trajectory_frame_dims_count(traj, ksp)
    (traj isa NamedDimsArray && image isa NamedDimsArray) || return 0
    s = dimnames(traj)[2:end]
    b = dimnames(image)[(nspatial + 1):end]
    for f in min(length(b), length(s) - 1):-1:1
        s[(end - f + 1):end] == b[(end - f + 1):end] && return f
    end
    return 0
end

"""
    _simulate_partitioned_acquisition(image, acq_info)

Simulation when the per-frame subsampling specs select different numbers of samples: the result
cannot be one dense array, so each frame is simulated on its own and the frames are collected into
a [`PartitionedKSpace`](@ref).

Each frame goes through the ordinary dense path — one frame, one spec — so the samples are exactly
what a per-frame acquisition would have produced, which is also what the partitioned encoding
operator computes for the whole series at once.
"""
function _simulate_partitioned_acquisition(image, acq_info::CartesianAcquisitionInfo)
    specs = acq_info.subsampling
    nd = ndims(image)
    spatial_dims = acq_info.is3D ? 3 : 2
    @argcheck nd > spatial_dims "image must have a trailing dimension for the per-frame subsampling specs"
    @argcheck size(image, nd) == length(specs) "the $(length(specs)) subsampling specs must span the image's last dimension (size $(size(image, nd)))"

    frame_names = image isa NamedDimsArray ? dimnames(image)[1:(end - 1)] : nothing
    raw_image = unname(image)
    # Copy constructor rather than a hand-written field list: every field but the two overridden
    # here is carried across, so a field added to the type later cannot be silently dropped.
    frame_acqs = map(specs) do spec
        return CartesianAcquisitionInfo(acq_info; subsampling = spec, kspace_data = nothing)
    end
    frame_ksps = map(enumerate(frame_acqs)) do (frame, frame_acq)
        frame_image = collect(selectdim(raw_image, nd, frame))
        if !isnothing(frame_names)
            frame_image = NamedDimsArray{frame_names}(frame_image)
        end
        return simulate_acquisition(frame_image, frame_acq; inverse_crime_check = false, keep_sensitivity_maps = false).kspace_data
    end
    ksp_names = if isnothing(frame_names)
        nothing
    else
        (dimnames(first(frame_ksps))..., dimnames(image)[end])
    end
    return PartitionedKSpace(
        collect(frame_ksps);
        ragged_dim = _ragged_subsampling_dim(acq_info.image_size, specs),
        dimnames = ksp_names,
    )
end

# ─── Simulation on a finer grid ─────────────────────────────────────────────────────────────────
#
# The phantom and the reconstruction grid cover the same field of view, voxel edge to voxel edge:
# voxel `j` of an `n`-voxel axis is centred at `(j - 1/2) / n` of the FOV. A Fourier operator
# places the origin of its phase at voxel `o` (`n ÷ 2 + 1` along a centred image axis, `1`
# otherwise), so the same object has k-space `D_n(k) ≈ F(k) e^{2πi k (o - 1/2)/n} / Δ_n` on an
# `n`-voxel grid. Data on the reconstruction grid (`M` voxels) follow from the phantom's (`N`) as
#
#     D_M(k) = D_N(k) · (M / N) · exp(2πi k ((o_M - 1/2)/M - (o_N - 1/2)/N)),
#
# per spatial axis, for every frequency `k` the reconstruction grid has: the samples of the finer
# grid outside that band are what the reconstruction cannot represent.
#
# Cartesian k-space is centred, its zero frequency at `cld(n, 2) + 1` (the position `ifftshift`
# moves it to), except along the axes listed in `shifted_kspace_dims`, which are in FFT order;
# the image origin is voxel 1 except along the axes listed in `shifted_image_dims`. The
# non-Cartesian operator always places it at the centre voxel.

# The image-domain origin of the Fourier operator along an axis of `n` voxels.
_phase_origin(n::Integer, centred::Bool) = centred ? n ÷ 2 + 1 : 1

# The signed frequency (cycles per FOV, in `-(n ÷ 2):((n - 1) ÷ 2)`) at position `p` of a
# k-space axis of `n` samples, centred or in FFT order.
function _axis_frequency(p::Integer, n::Integer, centred::Bool)
    q = centred ? mod(p - 1 - cld(n, 2), n) : p - 1
    return q <= (n - 1) ÷ 2 ? q : q - n
end

# The position of frequency `k` on such an axis.
_frequency_position(k::Integer, n::Integer, centred::Bool) = mod(k + (centred ? cld(n, 2) : 0), n) + 1

# The phase offset `δ` of the formula above, between grids of `M` and `N` voxels.
_origin_offset(M::Integer, N::Integer, centred::Bool) =
    (_phase_origin(M, centred) - 1 / 2) / M - (_phase_origin(N, centred) - 1 / 2) / N

# The k-space of the reconstruction grid from that of the phantom grid: along each axis in `dims`,
# keep the `M` frequencies the coarser grid has, shift their phase to its origin, and scale by
# `M / N`.
function _restrict_kspace(K::AbstractArray, dims, coarse::Tuple, centred_k, centred_img)
    out = K
    T = real(eltype(K))
    for (i, d) in enumerate(dims)
        N, M = size(K, d), coarse[i]
        N == M && continue
        ks = [_axis_frequency(p, M, centred_k[i]) for p in 1:M]
        src = [_frequency_position(k, N, centred_k[i]) for k in ks]
        δ = _origin_offset(M, N, centred_img[i])
        w = [T(M / N) * cispi(T(2 * k * δ)) for k in ks]
        idx = ntuple(j -> j == d ? _to_storage_of(out, src) : Colon(), ndims(out))
        shape = ntuple(j -> j == d ? M : 1, ndims(out))
        out = out[idx...] .* _to_storage_of(out, reshape(w, shape))
    end
    return out
end

# `acq` with its shifted axes given by position rather than by name, so that its copies without
# named k-space (the phantom-grid acquisition, the result of an unnamed phantom) stay valid.
function _with_positional_shifts(acq::CartesianAcquisitionInfo)
    ksp = something(acq.kspace_data, Int[])
    sk = _normalize_shifted_dims(acq.shifted_kspace_dims, acq.is3D, ksp, "shifted_kspace_dims", (:kx, :ky, :kz))
    si = _normalize_shifted_dims(acq.shifted_image_dims, acq.is3D, ksp, "shifted_image_dims", (:x, :y, :z))
    return CartesianAcquisitionInfo(acq; shifted_kspace_dims = Tuple(sk), shifted_image_dims = Tuple(si))
end

# The spatial size of the phantom: the grid the data are simulated on.
function _simulation_grid(phantom, acq_info::AcquisitionInfo)
    nspatial = acq_info.is3D ? 3 : 2
    @argcheck ndims(phantom) >= nspatial "the phantom must have at least $nspatial dimensions"
    return _leading_size(phantom, acq_info.image_size)
end

_leading_size(x, ::NTuple{N, Integer}) where {N} = ntuple(d -> size(x, d), Val(N))

function _check_simulation_grid(grid::Tuple, recon::Tuple, inverse_crime_check::Bool)
    all(grid .>= recon) || throw(
        ArgumentError(
            "the phantom ($(join(grid, "×"))) must be at least as large as the reconstruction " *
                "grid `image_size` ($(join(recon, "×"))) along every spatial axis",
        ),
    )
    inverse_crime_check || return nothing
    equal = findall(grid .== recon)
    multiple = findall((grid .% recon .== 0) .& (grid .!= recon))
    suggested = join(ifelse.(grid .== recon, round.(Int, 1.58 .* recon), grid), "×")
    if length(equal) == length(recon)
        @warn "simulate_acquisition: the phantom has the reconstruction's size " *
            "$(join(recon, "×")), so the data are simulated with the operator that will " *
            "reconstruct them (the inverse crime), and reconstructions look better than they " *
            "would on measured data. Simulate from a finer, area-sampled phantom, e.g. " *
            "$suggested; pass `inverse_crime_check = false` where the consistency is intended."
    elseif !isempty(equal)
        @warn "simulate_acquisition: along spatial axes $(join(equal, ", ")) the phantom has " *
            "the reconstruction's size, so along them the data are simulated with the operator " *
            "that will reconstruct them (the inverse crime). Make the phantom finer along every " *
            "axis, e.g. $suggested; pass `inverse_crime_check = false` where the consistency " *
            "is intended."
    end
    if !isempty(multiple)
        @warn "simulate_acquisition: along spatial axes $(join(multiple, ", ")) the phantom is " *
            "an integer multiple ($(join(grid[multiple] .÷ recon[multiple], "×"))) of the " *
            "reconstruction grid, so every voxel centre of the reconstruction is also one of " *
            "the phantom; a point-sampled phantom then gives optimistic errors. Prefer a " *
            "non-integer ratio or an area-sampled phantom; pass `inverse_crime_check = false` " *
            "to silence this."
    end
    return nothing
end

function _check_simulation_maps(acq_info::AcquisitionInfo, grid::Tuple)
    smaps = acq_info.sensitivity_maps
    isnothing(smaps) && return nothing
    size(smaps)[1:length(grid)] == grid || throw(
        ArgumentError(
            "the sensitivity maps ($(join(size(smaps)[1:length(grid)], "×"))) must have the " *
                "phantom's spatial size ($(join(grid, "×"))): the coils act on the phantom grid",
        ),
    )
    return nothing
end

# The maps the returned acquisition carries: none unless `keep` is `true`, else the simulation's
# maps on the reconstruction grid.
function _returned_maps(acq_info::AcquisitionInfo, grid::Tuple, keep::Union{Bool, Nothing})
    smaps = acq_info.sensitivity_maps
    isnothing(smaps) && return nothing
    if isnothing(keep)
        @info "simulate_acquisition: the returned acquisition carries no sensitivity maps, since " *
            "reusing the maps that simulated the data is part of the inverse crime. Estimate " *
            "them (`estimate_sensitivities`) before reconstructing, or pass " *
            "`keep_sensitivity_maps = true` to keep them (`false` drops them silently)."
    end
    keep === true || return nothing
    grid == acq_info.image_size && return smaps
    return _resample_sensitivity_maps(smaps, length(grid), acq_info.image_size)
end

# The sensitivity maps resampled to the reconstruction grid, by the same restriction of their
# spectrum as the data.
function _resample_sensitivity_maps(smaps, nspatial::Int, recon::Tuple)
    raw = unname(smaps)
    dims = 1:nspatial
    fft_order = ntuple(_ -> false, nspatial)
    out = ifft(_restrict_kspace(fft(raw, dims), dims, recon, fft_order, fft_order), dims)
    out = eltype(raw) <: Real ? real.(out) : convert.(eltype(raw), out)
    return smaps isa NamedDimsArray ? NamedDimsArray{dimnames(smaps)}(out) : out
end

# Cartesian data on the reconstruction grid from a finer phantom: the full k-space of the phantom
# grid, restricted to the reconstruction grid's frequencies, then subsampled.
function _simulate_on_finer_grid(phantom, acq_info::CartesianAcquisitionInfo, grid::Tuple)
    nspatial = length(grid)
    recon = acq_info.image_size
    fine = CartesianAcquisitionInfo(acq_info; image_size = grid, subsampling = nothing, kspace_data = nothing)
    K = _simulate_on_grid(phantom, fine)
    centred_k = ntuple(d -> !(d in acq_info.shifted_kspace_dims), nspatial)
    centred_img = ntuple(d -> d in acq_info.shifted_image_dims, nspatial)
    Kc = _restrict_kspace(unname(K), 1:nspatial, recon, centred_k, centred_img)
    K isa NamedDimsArray && (Kc = NamedDimsArray{dimnames(K)}(Kc))
    isnothing(acq_info.subsampling) && return Kc
    # The subsampled layout, and the operator that takes the full k-space to it, are those of the
    # reconstruction grid.
    coarse_image = similar(unname(phantom), (recon..., size(phantom)[(nspatial + 1):end]...))
    phantom isa NamedDimsArray && (coarse_image = NamedDimsArray{dimnames(phantom)}(coarse_image))
    ksp = _kspace_template(coarse_image, acq_info)
    ℳ = get_subsampling_operator(ksp, recon, acq_info.subsampling; threaded = false)
    mul!(ksp, ℳ, Kc)
    return ksp
end

# Non-Cartesian data from a finer phantom: the same physical frequencies, sampled on the phantom
# grid, where they are a smaller fraction of the grid's bandwidth.
function _simulate_on_finer_grid(phantom, acq_info::NonCartesianAcquisitionInfo, grid::Tuple)
    recon = acq_info.image_size
    nspatial = length(grid)
    traj = acq_info.trajectory
    ratio = reshape(collect(eltype(traj), recon ./ grid), nspatial, ntuple(_ -> 1, ndims(traj) - 1)...)
    fine_traj = unname(traj) .* _to_storage_of(unname(traj), ratio)
    traj isa NamedDimsArray && (fine_traj = NamedDimsArray{dimnames(traj)}(fine_traj))
    fine = NonCartesianAcquisitionInfo(acq_info; image_size = grid, trajectory = fine_traj)
    # An NFFT about 100 times more accurate than a reconstruction's, at the level of `Float32`
    # rounding, so the data do not carry the reconstruction's gridding error.
    K = _simulate_on_grid(phantom, fine; m = 4, sigma = 2.0)
    # The amplitude and phase that move the data to the reconstruction grid's voxel size and
    # origin, per sample: `(M / N) exp(2πi k δ)` along each axis, with `k = traj · M` in cycles
    # per FOV.
    T = real(eltype(K))
    δ = ntuple(d -> T(_origin_offset(recon[d], grid[d], true)), nspatial)
    amplitude = T(prod(recon ./ grid))
    host_traj = Array(unname(traj))
    w = map(CartesianIndices(size(host_traj)[2:end])) do I
        amplitude * cispi(2 * sum(T(host_traj[d, I]) * recon[d] * δ[d] for d in 1:nspatial))
    end
    nframe = _simulation_frame_dims_count(traj, phantom, nspatial, acq_info.kspace_data)
    out = unname(K) .* _broadcast_weights(unname(K), w, nframe)
    return K isa NamedDimsArray ? NamedDimsArray{dimnames(K)}(out) : out
end

function get_kspace_size(image, acq_info::CartesianAcquisitionInfo)
    if isnothing(acq_info.subsampling) && isnothing(acq_info.sensitivity_maps)
        return size(image)
    elseif acq_info.is3D
        @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 3D acquisition"
        transformed_size = get_transformed_size(image, acq_info)
        if !isnothing(acq_info.sensitivity_maps)
            return (transformed_size..., size(acq_info.sensitivity_maps, 4), size(image)[4:end]...)
        else
            return (transformed_size..., size(image)[4:end]...)
        end
    elseif !isnothing(acq_info.sensitivity_maps)
        if ndims(acq_info.sensitivity_maps) == 4
            @argcheck ndims(image) >= 3 "image must have at least 3 dimensions for 2D multislice acquisition"
        else
            @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
        end
        transformed_size = get_transformed_size(image, acq_info)
        return (transformed_size..., size(acq_info.sensitivity_maps, 3), size(image)[3:end]...)
    else
        @argcheck ndims(image) >= 2 "image must have at least 2 dimensions for 2D acquisition"
        transformed_size = get_transformed_size(image, acq_info)
        return (transformed_size..., size(image)[3:end]...)
    end
end

function get_transformed_size(image, acq_info::CartesianAcquisitionInfo)
    if isnothing(acq_info.subsampling)
        return acq_info.image_size
    elseif acq_info.subsampling isa AbstractArray
        spatial_dims = acq_info.is3D ? 3 : 2
        spreading_dims = ndims(acq_info.subsampling)
        @argcheck ndims(image) >= spatial_dims + spreading_dims "image must provide one trailing dimension per subsampling pattern dimension"

        first_index = first(CartesianIndices(acq_info.subsampling))
        sample_img = @view image[fill(:, spatial_dims)..., Tuple(first_index)..., fill(1, ndims(image) - spatial_dims - spreading_dims)...]
        sample_subsampling = acq_info.subsampling[first_index]
        Base.checkbounds(sample_img, sample_subsampling...)
        return size(@view(sample_img[sample_subsampling...]))
    else
        if acq_info.is3D
            single_img = @view image[:, :, :, ones(Int, ndims(image) - 3)...]
        else
            single_img = @view image[:, :, ones(Int, ndims(image) - 2)...]
        end
        Base.checkbounds(single_img, acq_info.subsampling...)
        return size(@view(single_img[acq_info.subsampling...]))
    end
end

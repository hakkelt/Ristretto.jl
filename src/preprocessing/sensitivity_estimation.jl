"""
    SensitivityEstimation

Abstract type representing coil sensitivity estimation algorithms.
"""
abstract type SensitivityEstimation end

"""
    SelfCalibrating(; calib_size = 24)

Direct low-resolution sensitivity estimation from the central auto-calibration signal (ACS) region
normalized by root-sum-of-squares (McKenzie et al. 2002, Bydder et al. 2002).
"""
struct SelfCalibrating{T} <: SensitivityEstimation
    calib_size::T
    function SelfCalibrating(; calib_size = 24)
        return new{typeof(calib_size)}(calib_size)
    end
end

"""
    AdaptiveCombine(; kernel_size = 5)

Adaptive coil sensitivity estimation and combination via local correlation matrix eigenanalysis (Walsh et al. 2000).
Needs no separate calibration region and produces smooth, SNR-optimal sensitivity maps.
"""
struct AdaptiveCombine{T} <: SensitivityEstimation
    kernel_size::T
    function AdaptiveCombine(; kernel_size = 5)
        return new{typeof(kernel_size)}(kernel_size)
    end
end

"""
    ESPIRiT(; calib_size = 24, kernel_size = 6, eigenvalue_threshold = 0.02, subspace_threshold = 0.05)

Eigenvalue-based sensitivity estimation (Uecker et al. 2014) from the central calibration subspace.
"""
struct ESPIRiT{T1, T2, T3, T4} <: SensitivityEstimation
    calib_size::T1
    kernel_size::T2
    eigenvalue_threshold::T3
    subspace_threshold::T4
    function ESPIRiT(;
            calib_size = 24,
            kernel_size = 6,
            eigenvalue_threshold = 0.02,
            subspace_threshold = 0.05,
        )
        return new{typeof(calib_size), typeof(kernel_size), typeof(eigenvalue_threshold), typeof(subspace_threshold)}(
            calib_size, kernel_size, eigenvalue_threshold, subspace_threshold
        )
    end
end

"""
    estimate_sensitivities(acq::CartesianAcquisitionInfo; method = SelfCalibrating())
    estimate_sensitivities(acq::NonCartesianAcquisitionInfo; method = SelfCalibrating(), dcf = acq.dcf, average_dims = (:time,), threaded = true)
    estimate_sensitivities(kspace::AbstractArray; method = SelfCalibrating(), is3D = false, coil_dim = nothing, image_size = nothing)

Estimates coil sensitivity maps from multi-coil k-space data using the specified method.
When passed an `AcquisitionInfo`, returns a new `AcquisitionInfo` with the `sensitivity_maps` field populated,
sized to match `acq.image_size` (the k-space is zero-padded, centered, if it only covers the measured extent
of a subsampled acquisition). Passing `image_size` explicitly has the same effect for the raw-array method.

K-space with batch dimensions past the coil axis (`:z` slices, `:time` frames, `:contrast`, ...)
is estimated slab by slab — coil sensitivities differ from slice to slice — and the maps come
back in the k-space's own layout, e.g. `(:x, :y, :coil, :z)` for multi-slice data.

## Non-Cartesian acquisitions

Every estimator here reads a calibration window out of a Cartesian grid, which non-Cartesian
samples are not. The `NonCartesianAcquisitionInfo` method therefore grids first: a
density-compensated NFFT adjoint gives one image per coil, a spatial FFT puts those back on a
Cartesian grid of `acq.image_size`, and the estimator runs on that. `dcf` is what weights the
gridding — `acq.dcf` when the acquisition carries one (vendor weights, or the output of
[`density_compensation`](@ref)), otherwise `:auto`, which lets NFFTOperators estimate it.
Gridding without any density compensation would hand the estimator a k-space centre weighted by
how densely the trajectory samples it, so `nothing` is rejected.

Batch dimensions are treated as for Cartesian data — one set of maps per slab — except for those
named in `average_dims` (`(:time,)` by default), which are averaged over before calibration — on
the samples, which the shared trajectory and the linearity of gridding make equivalent to
averaging the images, at one gridding pass instead of one per frame. A single frame of a
real-time or cine non-Cartesian series is usually far too
undersampled to calibrate from, while the coils do not move between frames, so the temporal mean
is both the better-conditioned and the physically correct calibration input. Pass
`average_dims = ()` to get one set of maps per frame instead; integers (indexing the gridded
image array) work in place of names, and are the only form available for unnamed k-space.

## FFT-shift convention

Every estimator inverts centered k-space into Ristretto's *default* image convention — image origin at
index 1, the plain-DFT one — so the raw-array method returns maps in that convention. An
acquisition that declares `shifted_image_dims` reconstructs **centered** images instead (scanner
data always does; see `AcquisitionInfo(::MRIBase.RawAcquisitionData)`), so the
`AcquisitionInfo` method `fftshift`s the maps onto those axes before attaching them. Without
that the maps are rolled by half the FOV relative to every image they multiply, and every
sensitivity-weighted reconstruction from real data is nonsense — not visibly shifted, just wrong.
Maps estimated by hand from a raw array must be shifted the same way before being attached to a
shifted acquisition.
"""
function estimate_sensitivities(
        acq::CartesianAcquisitionInfo;
        method::SensitivityEstimation = SelfCalibrating(),
    )
    _is_device(acq) && return _to_storage_of(acq, estimate_sensitivities(Adapt.adapt(Array, acq); method))
    _reject_partitioned(acq.kspace_data, "sensitivity estimation")
    is3D = acq.is3D
    sens = estimate_sensitivities(
        acq.kspace_data;
        method,
        is3D,
        image_size = acq.image_size,
    )
    if !isempty(acq.shifted_image_dims)
        sens = _shift_sensitivity_maps(sens, acq.shifted_image_dims, acq.kspace_data, is3D)
    end
    return AcquisitionInfo(acq; sensitivity_maps = sens)
end

function estimate_sensitivities(
        acq::NonCartesianAcquisitionInfo;
        method::SensitivityEstimation = SelfCalibrating(),
        dcf = isnothing(acq.dcf) ? :auto : acq.dcf,
        average_dims = (:time,),
        threaded::Bool = true,
    )
    if _is_device(acq)
        host = estimate_sensitivities(
            Adapt.adapt(Array, acq); method, dcf = _adapt_any(Array, dcf), average_dims, threaded
        )
        return _to_storage_of(acq, host)
    end
    @argcheck !isnothing(acq.kspace_data) "sensitivity estimation needs k-space data, and this NonCartesianAcquisitionInfo carries none"
    _reject_partitioned(acq.kspace_data, "sensitivity estimation")
    @argcheck !isnothing(dcf) "gridding for sensitivity estimation needs density compensation: pass `dcf = :auto` to estimate it, or attach one with `density_compensation`"

    is3D = acq.is3D
    nspatial = is3D ? 3 : 2
    spatial_dims = ntuple(identity, nspatial)

    # With a shared trajectory, averaging is done on the samples rather than on the gridded images:
    # gridding is linear, so the two are the same answer — but this way the NFFT adjoint runs once
    # instead of once per frame. A per-frame trajectory samples different k-space locations in each
    # frame, so there the frames are gridded one by one and the *images* are averaged.
    nfourier = _get_sample_dims_count(acq)
    nframe = _trajectory_frame_dims_count(acq.trajectory, acq.kspace_data)
    avg_idx = _calibration_average_indices(acq.kspace_data, average_dims, nfourier, nspatial)

    # One image per coil (and per remaining batch slab) from the density-compensated gridding
    # adjoint. The NFFT convention puts the image origin at the centre of the matrix, so the
    # k-space this produces below is centred too, and so are the maps that come out of it.
    averaged = if nframe == 0
        _grid_coil_images(_drop_mean(acq.kspace_data, avg_idx), acq, dcf, threaded)
    else
        # k-space position k is image position k - nfourier + nspatial: both layouts end in the
        # same `(coil, batch...)` tail.
        gridded = _grid_coil_images(acq.kspace_data, acq, dcf, threaded)
        _drop_mean(gridded, [i - nfourier + nspatial for i in avg_idx])
    end

    raw_images = unname(averaged)
    ksp_gridded = fftshift(fft(ifftshift(raw_images, spatial_dims), spatial_dims), spatial_dims)
    sens = _estimate_sensitivities_batched(ksp_gridded, method, nspatial + 1, is3D, nothing)
    sens = _shift_sensitivity_maps(sens, spatial_dims, ksp_gridded, is3D)

    if averaged isa NamedDimsArray
        sens = NamedDimsArray{dimnames(averaged)}(sens)
    end
    return AcquisitionInfo(acq; sensitivity_maps = sens)
end

# The gridding adjoint, built from the acquisition's own trajectory with the density compensation
# the caller chose. Symbol dimension names on the trajectory are only meaningful when the k-space
# carries them too, so an unnamed k-space grids through the plain-array operator.
function _grid_coil_images(ksp::AbstractArray, acq::NonCartesianAcquisitionInfo, dcf, threaded::Bool)
    return if ksp isa NamedDimsArray
        𝒩 = get_fourier_operator(ksp, acq.image_size, acq.trajectory; dcf, threaded)
        𝒩' * ksp
    else
        traj = acq.trajectory isa NamedDimsArray ? unname(acq.trajectory) : acq.trajectory
        raw_dcf = dcf isa NamedDimsArray ? unname(dcf) : dcf
        𝒩 = get_fourier_operator(ksp, acq.image_size, traj; dcf = raw_dcf, threaded)
        𝒩' * ksp
    end
end

"""
    _calibration_average_indices(kspace, average_dims, nfourier, nspatial) -> Vector{Int}

The k-space positions of the batch dimensions named (or indexed) in `average_dims`, over which
sensitivity estimation averages. Names are resolved against the k-space's own dimension names and
silently ignored when the array carries none or does not have that dimension — `(:time,)`, the
default, must be a no-op for the many acquisitions that have no time axis. Integers index the
*gridded image* array the caller sees (`(:x, :y, :coil, batch...)`), which differs from the
k-space layout whenever the trajectory's sample axes do not number `nspatial`. Sample axes and
the coil axis are never averaged.
"""
function _calibration_average_indices(kspace::AbstractArray, average_dims, nfourier::Int, nspatial::Int)
    dims = average_dims isa Union{Integer, Symbol} ? (average_dims,) : average_dims
    names = kspace isa NamedDimsArray ? dimnames(kspace) : ()
    idx = Int[]
    for d in dims
        # `average_dims` is untyped, so without this the resolved index stays `Any` and
        # `i in idx` widens to every `in` method in scope, including the term-building one.
        i::Union{Nothing, Int} = if d isa Integer
            # From an image-array position to the matching k-space position: both layouts end in
            # the same `(coil, batch...)` tail, they only differ in how many axes come before it.
            k = Int(d) - nspatial + nfourier
            k <= ndims(kspace) ? k : nothing
        else
            @argcheck d isa Symbol "average_dims entries must be Integer or Symbol, got $d"
            isempty(names) ? nothing : findfirst(==(d), names)
        end
        if isnothing(i)
            # A name the k-space does not carry is a no-op (`:time` on data that has no time
            # axis), unless it is a spatial image axis, which is never something to average over.
            @argcheck d ∉ (:x, :y, :z) "average_dims names $d, a spatial image dimension, not a batch dimension"
            continue
        end
        @argcheck i > nfourier + 1 "average_dims names dimension $d, which is a sample or coil dimension of the k-space, not a batch dimension"
        i in idx || push!(idx, i)
    end
    return idx
end

# Mean over the dimensions `idx` of `a`, dropping them (and their names).
function _drop_mean(a::AbstractArray, idx)
    isempty(idx) && return a
    averaged = dropdims(mean(unname(a), dims = Tuple(idx)), dims = Tuple(idx))
    return if a isa NamedDimsArray
        NamedDimsArray{Tuple(n for (i, n) in enumerate(dimnames(a)) if i ∉ idx)}(averaged)
    else
        averaged
    end
end

# `shifted_image_dims` names *spatial* image axes (`:x`, `:y`, `:z`, or 1/2/3); the maps carry
# those axes in the same order as the k-space they were estimated from, with the coil axis
# wherever it sat there. Resolve the one to the other, then `fftshift`.
function _shift_sensitivity_maps(sens, shifted_image_dims, kspace, is3D::Bool)
    spatial_indices = _normalize_shifted_dims(
        shifted_image_dims, is3D, kspace, "shifted_image_dims", (:x, :y, :z)
    )
    c_idx = _resolve_coil_dim(sens, nothing; fallback = is3D ? 4 : 3)
    # Only the leading `is3D ? 3 : 2` non-coil axes are spatial; anything past them is a batch
    # dimension (slices, frames, contrasts) and is never fftshifted.
    spatial_axes = [i for i in 1:ndims(sens) if i != c_idx][1:(is3D ? 3 : 2)]
    axes_to_shift = Tuple(spatial_axes[i] for i in spatial_indices)
    shifted = fftshift(unname(sens), axes_to_shift)
    return sens isa NamedDimsArray ? NamedDimsArray{dimnames(sens)}(shifted) : shifted
end

function estimate_sensitivities(
        kspace::AbstractArray;
        method::SensitivityEstimation = SelfCalibrating(),
        is3D::Bool = false,
        coil_dim = nothing,
        image_size = nothing,
    )
    _is_device(kspace) &&
        return _on_host(k -> estimate_sensitivities(k; method, is3D, coil_dim, image_size), kspace, kspace)
    c_idx = _resolve_coil_dim(kspace, coil_dim; fallback = is3D ? 4 : 3)

    raw_ksp = unname(kspace)
    sens_arr = _estimate_sensitivities_batched(raw_ksp, method, c_idx, is3D, image_size)

    if kspace isa NamedDimsArray
        k_dims = dimnames(kspace)
        img_dims = ntuple(length(k_dims)) do i
            d = k_dims[i]
            d == :kx ? :x : (d == :ky ? :y : (d == :kz ? :z : d))
        end
        return NamedDimsArray{img_dims}(sens_arr)
    else
        return sens_arr
    end
end

"""
    _estimate_sensitivities_batched(kspace, method, c_idx, is3D, image_size)

Estimate maps for k-space that carries batch dimensions (slices, frames, contrasts, ...) beyond
the `is3D ? 3 : 2` spatial axes and the coil axis: every slab gets its own maps, since coil
sensitivities differ from slice to slice, and the results are stacked back into the batch layout
the k-space had. K-space without batch dimensions goes straight to the estimator.
"""
function _estimate_sensitivities_batched(kspace::AbstractArray, method, c_idx, is3D::Bool, image_size)
    function estimate(slab)
        slab_maps = _estimate_sensitivities_core(
            isnothing(image_size) ? slab : _pad_kspace_to_image_size(slab, image_size, c_idx),
            method, c_idx, is3D,
        )
        if iszero(slab_maps) && !iszero(slab)
            # All-zero maps are not an estimate, they are a silent wrong answer: every
            # sensitivity-weighted reconstruction that uses them comes out zero, and dividing by
            # them (as the iterative solvers do) produces NaNs several stages later.
            @warn "Sensitivity estimation produced all-zero maps from k-space that is not " *
                "empty: the calibration region at the centre of the encoded matrix holds no " *
                "signal. Either the k-space centre is not where the header says it is — check " *
                "`head.center_sample` and the encoding limits of the file it came from — or a " *
                "spatial axis is too short for the calibration region to fit, which is what a " *
                "3D acquisition holding a single partition looks like (reconstruct that one as " *
                "2D). Otherwise, pass maps estimated by hand." maxlog = 1
        end
        return slab_maps
    end

    nspatial = is3D ? 3 : 2
    nbatch = ndims(kspace) - nspatial - 1
    nbatch <= 0 && return estimate(kspace)

    @argcheck c_idx == nspatial + 1 "batch dimensions are only supported after the coil dimension: coil is dimension $(c_idx) of $(ndims(kspace)), expected $(nspatial + 1)"
    batch_sizes = size(kspace)[(nspatial + 2):end]
    lead = ntuple(_ -> Colon(), nspatial + 1)

    maps = nothing
    for b in CartesianIndices(batch_sizes)
        slab = kspace[lead..., Tuple(b)...]
        s = estimate(slab)
        if maps === nothing
            maps = similar(s, size(s)..., batch_sizes...)
        end
        maps[lead..., Tuple(b)...] = s
    end
    return maps
end

"""
    _pad_kspace_to_image_size(kspace, image_size, c_idx)

Zero-pad `kspace`, centered, so its spatial dimensions (all dimensions but `c_idx`) match
`image_size`. A no-op when they already match. Assumes `kspace`'s own measured extent is
already centered on its k-space DC, as `_estimate_sensitivities_core` does when locating the
calibration window.
"""
function _pad_kspace_to_image_size(kspace::AbstractArray{T, N}, image_size, c_idx) where {T, N}
    spatial_idx = [i for i in 1:N if i != c_idx]
    @argcheck length(spatial_idx) == length(image_size) "image_size length must match the number of spatial dimensions"
    current = size(kspace)
    all(current[i] == image_size[j] for (j, i) in enumerate(spatial_idx)) && return kspace

    target = collect(current)
    for (j, i) in enumerate(spatial_idx)
        target[i] = image_size[j]
    end
    padded = zeros(T, target...)
    ranges = ntuple(N) do i
        if i in spatial_idx
            cur, tgt = current[i], target[i]
            start = (tgt - cur) ÷ 2 + 1
            start:(start + cur - 1)
        else
            1:current[i]
        end
    end
    padded[ranges...] = kspace
    return padded
end

# Core algorithm implementations

function _estimate_sensitivities_core(
        kspace::AbstractArray{T, N},
        method::SelfCalibrating,
        c_idx::Int,
        is3D::Bool,
    ) where {T, N}
    # Move the coil axis to the trailing position; `inv_perm` restores the layout.
    perm = _trailing_perm(c_idx, N)
    inv_perm = _trailing_inv_perm(c_idx, N)
    ksp_trailing = permutedims(kspace, perm)

    spatial_dims = size(ksp_trailing)[1:(N - 1)]
    Nc = size(ksp_trailing, N)
    cal_size = method.calib_size

    # Windowed calibration region
    cal_ksp = zeros(complex(T), size(ksp_trailing))
    ranges = ntuple(length(spatial_dims)) do d
        K_d = cal_size isa Tuple ? cal_size[d] : min(spatial_dims[d], cal_size)
        c_d = spatial_dims[d] ÷ 2 + 1
        (c_d - K_d ÷ 2):(c_d + K_d ÷ 2 - 1)
    end

    W = ones(real(T), map(length, ranges))
    for d in 1:length(spatial_dims)
        Kd = length(ranges[d])
        wd = 0.5 .- 0.5 .* cos.(2.0 * π .* (0:(Kd - 1)) ./ max(1, Kd - 1))
        w_shape = ntuple(i -> i == d ? Kd : 1, length(spatial_dims))
        W .*= reshape(wd, w_shape)
    end

    for c in 1:Nc
        cal_ksp[ranges..., c] = ksp_trailing[ranges..., c] .* W
    end

    # IFFT from centered k-space to uncentered image space
    lowres_img = zeros(complex(T), size(ksp_trailing))
    f_dims = ntuple(identity, length(spatial_dims))
    ℱ = _axis_dft_op(zeros(complex(T), spatial_dims...), f_dims; kspace_shift = true)
    for c in 1:Nc
        cal_c = selectdim(cal_ksp, N, c)
        selectdim(lowres_img, N, c) .= (ℱ' * collect(cal_c)) .* sqrt(prod(spatial_dims))
    end

    rss = sqrt.(sum(abs2.(lowres_img), dims = N))
    sens_trailing = lowres_img ./ (rss .+ eps(real(T)))
    return permutedims(sens_trailing, inv_perm)
end

function _estimate_sensitivities_core(
        kspace::AbstractArray{T, N},
        method::AdaptiveCombine,
        c_idx::Int,
        is3D::Bool,
    ) where {T, N}
    # Move the coil axis to the trailing position; `inv_perm` restores the layout.
    perm = _trailing_perm(c_idx, N)
    inv_perm = _trailing_inv_perm(c_idx, N)
    ksp_trailing = permutedims(kspace, perm)

    spatial_dims = size(ksp_trailing)[1:(N - 1)]
    Nc = size(ksp_trailing, N)
    f_dims = ntuple(identity, length(spatial_dims))

    coil_imgs = zeros(complex(T), size(ksp_trailing))
    ℱ = _axis_dft_op(zeros(complex(T), spatial_dims...), f_dims; kspace_shift = true)
    for c in 1:Nc
        ksp_c = selectdim(ksp_trailing, N, c)
        selectdim(coil_imgs, N, c) .= (ℱ' * collect(ksp_c)) .* sqrt(prod(spatial_dims))
    end

    K = method.kernel_size
    pad = K ÷ 2
    sens_trailing = zeros(complex(T), size(ksp_trailing))

    for idx in CartesianIndices(spatial_dims)
        patch_ranges = ntuple(length(spatial_dims)) do d
            max(1, idx[d] - pad):min(spatial_dims[d], idx[d] + pad)
        end
        patch = coil_imgs[patch_ranges..., :]
        patch_mat = reshape(permutedims(patch, (N, 1:(N - 1)...)), Nc, :)
        R = patch_mat * patch_mat'

        F = eigen(Hermitian(R))
        v = F.vectors[:, end]
        if abs(v[1]) > 1.0e-6
            v .*= cis(-angle(v[1]))
        end
        sens_trailing[idx, :] = v
    end

    return permutedims(sens_trailing, inv_perm)
end

function _estimate_sensitivities_core(
        kspace::AbstractArray{T, N},
        method::ESPIRiT,
        c_idx::Int,
        is3D::Bool,
    ) where {T, N}
    # Move the coil axis to the trailing position; `inv_perm` restores the layout.
    perm = _trailing_perm(c_idx, N)
    inv_perm = _trailing_inv_perm(c_idx, N)
    ksp_trailing = permutedims(kspace, perm)

    spatial_dims = size(ksp_trailing)[1:(N - 1)]
    Nc = size(ksp_trailing, N)
    cal_size = method.calib_size
    K = method.kernel_size

    ranges = ntuple(length(spatial_dims)) do d
        K_d = cal_size isa Tuple ? cal_size[d] : min(spatial_dims[d], cal_size)
        c_d = spatial_dims[d] ÷ 2 + 1
        (c_d - K_d ÷ 2):(c_d + K_d ÷ 2 - 1)
    end
    calib = ksp_trailing[ranges..., :]

    cal_dims = map(length, ranges)
    kernel_dims = ntuple(i -> K isa Tuple ? K[i] : K, length(spatial_dims))
    num_patches_per_dim = ntuple(i -> cal_dims[i] - kernel_dims[i] + 1, length(spatial_dims))
    num_patches = prod(num_patches_per_dim)
    patch_dim = prod(kernel_dims) * Nc

    C = zeros(complex(T), num_patches, patch_dim)
    p_idx = 1
    for p_offset in CartesianIndices(num_patches_per_dim)
        patch_box = ntuple(i -> (p_offset[i]):(p_offset[i] + kernel_dims[i] - 1), length(spatial_dims))
        patch = calib[patch_box..., :]
        C[p_idx, :] = reshape(patch, :)
        p_idx += 1
    end

    F_svd = svd(C)
    s_thresh = F_svd.S[1] * method.subspace_threshold
    num_vecs = count(s -> s >= s_thresh, F_svd.S)
    num_vecs = max(1, num_vecs)
    V_sub = F_svd.V[:, 1:num_vecs]

    f_dims = ntuple(identity, length(spatial_dims))
    ℱ = _axis_dft_op(zeros(complex(T), spatial_dims...), f_dims)
    V_img = zeros(complex(T), spatial_dims..., Nc, num_vecs)
    for v_idx in 1:num_vecs
        kernel_arr = reshape(V_sub[:, v_idx], kernel_dims..., Nc)
        flipped = reverse(conj(kernel_arr), dims = f_dims)
        for c in 1:Nc
            padded = zeros(complex(T), spatial_dims...)
            init_ranges = ntuple(i -> 1:kernel_dims[i], length(spatial_dims))
            padded[init_ranges...] = flipped[init_ranges..., c]
            shift_amounts = ntuple(i -> -(kernel_dims[i] ÷ 2), length(spatial_dims))
            padded = circshift(padded, shift_amounts)
            V_img[fill(:, length(spatial_dims))..., c, v_idx] .= (ℱ * padded) .* sqrt(prod(spatial_dims))
        end
    end

    sens_trailing = zeros(complex(T), size(ksp_trailing))
    for idx in CartesianIndices(spatial_dims)
        V_r = reshape(V_img[idx, :, :], Nc, num_vecs)
        W = V_r * V_r'
        F_eig = eigen(Hermitian(W))
        val = F_eig.values[end]
        vec_max = F_eig.vectors[:, end]
        if val >= method.eigenvalue_threshold
            # An eigenvector is only defined up to a phase, and LAPACK's choice of it varies from
            # pixel to pixel and from run to run, which would leave the maps — and the phase of
            # every image reconstructed with them — arbitrary. Reference the phase to the first
            # coil, as `AdaptiveCombine` does.
            if abs(vec_max[1]) > 1.0e-6
                vec_max = vec_max .* cis(-angle(vec_max[1]))
            end
            sens_trailing[idx, :] = vec_max
        end
    end

    return permutedims(sens_trailing, inv_perm)
end

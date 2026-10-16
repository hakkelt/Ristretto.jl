function get_slices(plan, acq_data)
    indices = CartesianIndices(plan.kspace_size[collect(plan.kspace_batch_dims)])
    get_index_max_width = d -> length(string(size(acq_data.kspace_data, d)))
    slice_idx_widths = map(get_index_max_width, plan.kspace_batch_dims)
    slice_ids = map(idx -> get_slice_id(plan, idx, slice_idx_widths), indices)
    slices = _ksp_eachslice(acq_data.kspace_data, plan.kspace_batch_dims)
    ssm = plan.slices_sensitivity_maps
    local_acq = (
        get_acquisition_info_slice(acq_data, plan, idx, ksp_slice, ssm) for
            (idx, ksp_slice) in zip(indices, slices)
    )
    return zip(indices, slice_ids, local_acq)
end

function get_acquisition_info_slice(acq_info::CartesianAcquisitionInfo, plan, idx, kspace_data, slice_sensitivity_maps)
    subsampling = slice_subsampling(acq_info.subsampling, acq_info.kspace_data, plan.kspace_batch_dims, idx)
    sensitivity_maps = slice_sensitivity_maps_of(acq_info, idx, slice_sensitivity_maps)
    return CartesianAcquisitionInfo(acq_info; kspace_data, sensitivity_maps, subsampling)
end

function get_acquisition_info_slice(acq_info::NonCartesianAcquisitionInfo, plan, idx, kspace_data, slice_sensitivity_maps)
    ksp = acq_info.kspace_data
    nframe = _trajectory_frame_dims_count(acq_info.trajectory, ksp)
    trajectory = slice_frames(acq_info.trajectory, nframe, ndims(ksp), plan.kspace_batch_dims, idx)
    dcf = slice_frames(acq_info.dcf, nframe, ndims(ksp), plan.kspace_batch_dims, idx)
    sensitivity_maps = slice_sensitivity_maps_of(acq_info, idx, slice_sensitivity_maps)
    return NonCartesianAcquisitionInfo(acq_info; kspace_data, trajectory, dcf, sensitivity_maps)
end

# Maps with a slice axis (4-D for a 2-D encoding) go to each task one slice at a time.
slice_sensitivity_maps_of(acq_info, idx, slice_sensitivity_maps) =
    slice_sensitivity_maps ? @view(acq_info.sensitivity_maps[:, :, :, idx[1]]) : acq_info.sensitivity_maps

# The last `nframe` axes of a per-frame trajectory or dcf `a` are the last `nframe` axes of an
# `ksp_ndims`-dimensional k-space; each one a task splits is indexed down to that task's frame, as a
# copy, since an NFFT plan takes its nodes as a dense matrix. A shared trajectory (`nframe == 0`), or
# one whose frame axes the split leaves whole, is returned as is.
function slice_frames(a, nframe, ksp_ndims, kspace_batch_dims, idx)
    (isnothing(a) || nframe == 0) && return a
    slicer = ntuple(ndims(a)) do j
        j <= ndims(a) - nframe && return Colon()
        position = findfirst(==(ksp_ndims - (ndims(a) - j)), kspace_batch_dims)
        return isnothing(position) ? Colon() : idx[position]
    end
    all(s -> s isa Colon, slicer) && return a
    return a[slicer...]
end

# A subsampling spec shared by every batch element (`nothing`, or the per-axis tuple `(:, mask)`)
# is handed to each task unchanged.
slice_subsampling(subsampling, kspace_data, kspace_batch_dims, idx) = subsampling
slice_subsampling(subsampling::AbstractArray{Bool}, kspace_data, kspace_batch_dims, idx) = subsampling

# ... but `subsampling` may instead be an *array of specs*, one per batch element — a different ky
# mask per frame, say. Each task must then get the specs belonging to its own slice, not the whole
# array; otherwise the k-space it holds and the pattern that produced it no longer agree.
function slice_subsampling(subsampling::AbstractArray, kspace_data, kspace_batch_dims, idx)
    # Which k-space dimensions the spec array spans: the same alignment `get_subsampling_operator`
    # performs, against the non-Fourier k-space dimensions (coil and batch).
    fourier_dims = _get_subsampled_dims_count(subsampling)
    nonfourier = _ksp_trailing_size(kspace_data, fourier_dims + 1)
    n = ndims(subsampling)
    starts = filter(s -> nonfourier[s:(s + n - 1)] == size(subsampling), 1:(length(nonfourier) - n + 1))
    isempty(starts) && return subsampling
    # The size alone is ambiguous when another axis has the same length (8 coils and 8 frames):
    # the specs belong to batch elements, so an alignment that spans a batch dimension wins.
    spans_batch(s) = any(j -> fourier_dims + s - 1 + j in kspace_batch_dims, 1:n)
    i = findfirst(spans_batch, starts)
    start = isnothing(i) ? first(starts) : starts[i]
    slicer = ntuple(n) do j
        kspace_dim = fourier_dims + start - 1 + j
        position = findfirst(==(kspace_dim), kspace_batch_dims)
        return isnothing(position) ? Colon() : idx[position]
    end
    sliced = subsampling[slicer...]
    return sliced isa AbstractArray ? sliced : _normalize_subsampling(sliced)
end

function slice_x₀_components(x₀::Tuple, plan, idx)
    return map(x -> get_x₀_slice(x, plan, idx), x₀)
end

function slice_x₀_components(x₀::NamedTuple, plan, idx)
    return NamedTuple{keys(x₀)}(map(x -> get_x₀_slice(x, plan, idx), values(x₀)))
end

function get_x₀_slice(x₀, plan, idx)
    slicer = ntuple(length(plan.variable_size)) do d
        i = findfirst(==(d), plan.variable_batch_dims)
        isnothing(i) ? Colon() : idx[i]
    end
    return @view unname(x₀)[slicer...]
end

"""
    NonCartesianAcquisitionInfo(
            kspace_data;
            trajectory,
            dcf=nothing,
            sensitivity_maps=nothing,
            image_size,
            shifted_kspace_dims::Tuple=(),
            shifted_image_dims::Tuple=(),
            header=nothing,
    )

Container for non-Cartesian MRI acquisition settings.

`header` is the metadata: a [`Header`](@ref), stored as given, or keywords for one as a
`NamedTuple` or another dictionary. An empty header is created when none is given.

The trajectory stores coordinate axes in its first dimension. Its remaining
dimensions must match the non-coil k-space sample layout.

# Per-frame trajectories

A trajectory may also carry trailing *frame* axes, one trajectory per frame: a golden-angle
radial series whose spokes rotate from frame to frame is `(:coord, :sample, :spoke, :time)` for
k-space `(:sample, :spoke, :coil, :time)`. A trajectory axis after the sample axes is a frame axis
when it matches the trailing k-space axes — by name for a `NamedDimsArray`, by size otherwise —
and each frame is then encoded with its own NFFT (see [`get_fourier_operator`](@ref)). A
trajectory without frame axes is shared by every batch element, as before.

For plain arrays the match is by size alone, so a trajectory whose trailing size also matches the
axis right after the samples (a coil count equal to the frame count, or data without a coil axis)
is read as sharing one trajectory over more sample axes, which is what such a trajectory always
meant. Name the axes to make a per-frame trajectory unambiguous there.

`dcf` is forwarded to `NFFTOp` unchanged and follows its contract: `nothing` (the default)
applies **no** density compensation, so the encoding operator's adjoint is the true adjoint; an
array matching the trajectory sample dimensions (and its frame dimensions, for a per-frame
trajectory) makes that adjoint a density-compensated approximate inverse — the gridding
reconstruction — instead. Compute one with [`density_compensation`](@ref), which returns a copy of
the acquisition carrying it.
"""
struct NonCartesianAcquisitionInfo{K, T, D, S, I, SD, ID} <: AcquisitionInfo
    kspace_data::K
    trajectory::T
    dcf::D
    sensitivity_maps::S
    image_size::I
    shifted_kspace_dims::SD
    shifted_image_dims::ID
    is3D::Bool
    header::Header

    function NonCartesianAcquisitionInfo(ksp, traj, dcf, smaps, img_size, sK, sI, hdr = nothing)
        @argcheck !isnothing(traj) "trajectory must be provided"
        nd = ndims(traj)
        @argcheck nd > 1 "trajectory must have at least 2 dimensions"
        coord_dim = size(traj, 1)
        @argcheck coord_dim == 2 || coord_dim == 3 "the first dimension of trajectory must be 2 or 3"
        is3D = coord_dim == 3

        if traj isa NamedDimsArray
            first_name = dimnames(traj)[1]
            @argcheck first_name == :dim || first_name == :coord "trajectory first dimension must be :dim or :coord"
        end

        @argcheck !isnothing(img_size) "image_size must be provided"
        @argcheck length(img_size) == (is3D ? 3 : 2) "image_size length must match trajectory dimensionality"

        if !isnothing(ksp)
            if ksp isa NamedDimsArray
                @argcheck traj isa NamedDimsArray "trajectory must be a NamedDimsArray when k-space data is a NamedDimsArray"
            end
            nframe = _trajectory_frame_dims_count(traj, ksp)
            fourier_dims = ndims(traj) - 1 - nframe
            # Compared axis by axis: `fourier_dims` is only known at run time, and `==` on tuples of
            # unknown length infers as `Union{Missing, Bool}`, a runtime dispatch.
            @argcheck ndims(ksp) >= fourier_dims && all(i -> size(ksp, i) == size(traj, i + 1), 1:fourier_dims) "k-space data dimensions must match trajectory sample dimensions"
            if ksp isa NamedDimsArray
                @argcheck all(i -> dimnames(ksp, i) === dimnames(traj, i + 1), 1:fourier_dims) "k-space data dimension names must match trajectory sample dimension names"
                @argcheck :coil ∉ dimnames(traj) "a trajectory frame axis cannot be the :coil axis"
                if !isnothing(smaps)
                    @argcheck :coil ∈ dimnames(ksp) "k-space must have :coil dimension when sensitivity maps are provided"
                end
            end
        end

        if !isnothing(dcf)
            @argcheck size(dcf) == size(traj)[2:end] "dcf shape must match trajectory sample dimensions"
            @argcheck eltype(dcf) <: Real "dcf must be real-valued"
            @argcheck eltype(dcf) == eltype(traj) "dcf element type must match trajectory element type"
            if traj isa NamedDimsArray && dcf isa NamedDimsArray
                @argcheck dimnames(dcf) == dimnames(traj)[2:end] "dcf dimension names must match trajectory sample dimension names"
            end
        end

        if !isnothing(ksp)
            @argcheck eltype(ksp) == Complex{eltype(traj)} "k-space element type must match complex(eltype(trajectory))"
        end

        if !isnothing(smaps) && !isnothing(ksp)
            @argcheck eltype(ksp) == eltype(smaps) "k-space and sensitivity maps eltype mismatch"
            if is3D
                @argcheck ndims(smaps) == 4 "sensitivity maps must be 4D for 3D acquisition"
                @argcheck size(smaps)[1:3] == img_size "sensitivity maps spatial size must match image_size"
            elseif ndims(smaps) == 4
                @argcheck size(smaps)[1:2] == img_size "sensitivity maps spatial size must match image_size"
            else
                @argcheck ndims(smaps) == 3 "sensitivity maps must be 3D for 2D acquisition"
                @argcheck size(smaps)[1:2] == img_size "sensitivity maps spatial size must match image_size"
            end
        end

        _check_same_storage(ksp, smaps, "sensitivity maps")
        _check_same_storage(ksp, dcf, "dcf")
        hdr = _to_header(hdr)
        _check_geometry(hdr, img_size)
        return new{typeof(ksp), typeof(traj), typeof(dcf), typeof(smaps), typeof(img_size), typeof(sK), typeof(sI)}(
            ksp, traj, dcf, smaps, img_size, sK, sI, is3D, hdr
        )
    end
end

NonCartesianAcquisitionInfo(
    kspace_data;
    trajectory = nothing,
    dcf = nothing,
    sensitivity_maps = nothing,
    image_size = nothing,
    shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
    shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
    header = nothing,
) = NonCartesianAcquisitionInfo(kspace_data, trajectory, dcf, sensitivity_maps, image_size, shifted_kspace_dims, shifted_image_dims, header)

NonCartesianAcquisitionInfo(;
    kspace_data = nothing,
    trajectory = nothing,
    dcf = nothing,
    sensitivity_maps = nothing,
    image_size = nothing,
    shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
    shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
    header = nothing,
) = NonCartesianAcquisitionInfo(kspace_data, trajectory, dcf, sensitivity_maps, image_size, shifted_kspace_dims, shifted_image_dims, header)

"""
    _trajectory_frame_dims_count(trajectory, kspace) -> Int

How many trailing trajectory axes are *frame* axes, one trajectory per frame (see
[`NonCartesianAcquisitionInfo`](@ref)): `0` for a trajectory shared by every batch element, which
is also the answer when there is no k-space to compare against. Axes are compared by name when both
arrays are `NamedDimsArray`s and by size otherwise.
"""
function _trajectory_frame_dims_count(traj, ksp)
    isnothing(ksp) && return 0
    if traj isa NamedDimsArray && ksp isa NamedDimsArray
        return _frame_dims_count(dimnames(traj)[2:end], dimnames(ksp))
    end
    return _frame_dims_count(size(traj)[2:end], size(ksp))
end

# `s` describes the trajectory's non-coordinate axes and `k` the k-space's, as names or sizes. The
# shared reading — every trajectory axis is a sample axis — wins whenever it fits, so a trajectory
# that meant that before per-frame trajectories existed still means it. Otherwise the trajectory is
# `(samples..., frames...)` with the samples leading the k-space and the frames ending it; `0` when
# neither fits, leaving the constructor's shape check to report the mismatch.
function _frame_dims_count(s::Tuple, k::Tuple)
    n = length(s)
    (length(k) >= n && k[1:n] == s) && return 0
    for f in 1:(n - 1)
        nsample = n - f
        length(k) - f >= nsample || continue
        (k[1:nsample] == s[1:nsample] && k[(end - f + 1):end] == s[(nsample + 1):end]) && return f
    end
    return 0
end

"""
    _trajectory_sample_dims_count(trajectory, kspace) -> Int

Number of trajectory axes that index samples within one frame: every non-coordinate axis of a
shared trajectory, all but the frame axes of a per-frame one.
"""
_trajectory_sample_dims_count(traj, ksp) = ndims(traj) - 1 - _trajectory_frame_dims_count(traj, ksp)

function _get_acq_info_meta(info::NonCartesianAcquisitionInfo)
    meta = String[]
    if !isnothing(info.kspace_data)
        push!(meta, "kspace_data=Array{$(eltype(info.kspace_data))}<$(join(size(info.kspace_data), "×"))>")
    end
    push!(meta, "trajectory=Array{$(eltype(info.trajectory))}<$(join(size(info.trajectory), "×"))>")
    if !isnothing(info.dcf)
        push!(meta, "dcf=Array{$(eltype(info.dcf))}<$(join(size(info.dcf), "×"))>")
    end
    push!(meta, "encoding=" * (info.is3D ? "3D" : "2D"))
    push!(meta, "image_size=$(join(info.image_size, "×"))")
    if !isnothing(info.sensitivity_maps)
        push!(meta, "sensitivity_maps=$(eltype(info.sensitivity_maps))<$(join(size(info.sensitivity_maps), "×"))>")
    end
    return meta
end

function Base.show(io::IO, info::NonCartesianAcquisitionInfo)
    meta = _get_acq_info_meta(info)
    return print(io, "NonCartesianAcquisitionInfo(", join(meta, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", info::NonCartesianAcquisitionInfo)
    meta = _get_acq_info_meta(info)
    println(io, "NonCartesianAcquisitionInfo:")
    for (i, m) in enumerate(meta)
        m = replace(m, "=" => " = ", "_" => " ")
        print(io, "  - $m")
        if i < length(meta)
            println(io)
        end
    end
    return nothing
end

function get_subsampling_operator(::NonCartesianAcquisitionInfo; threaded::Bool = true)
    error("Subsampling operator is not applicable to non-Cartesian trajectories")
end

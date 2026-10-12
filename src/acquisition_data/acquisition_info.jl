"""
Abstract type for MRI acquisition information.

Subtypes:
- `CartesianAcquisitionInfo`: Cartesian (regular grid) acquisitions
- `NonCartesianAcquisitionInfo`: Non-Cartesian (trajectory-based) acquisitions
"""
abstract type AcquisitionInfo end

"""
    AcquisitionInfo(
            kspace_data=nothing;
            trajectory=nothing,
            is3D::Union{Bool,Nothing}=nothing,
            sensitivity_maps=nothing,
            image_size=nothing,
            subsampling=nothing,
            dcf=nothing,
            shifted_kspace_dims::Union{Tuple,Integer,Symbol}=(),
            shifted_image_dims::Union{Tuple,Integer,Symbol}=(),
            header=nothing,
    )

Smart constructor that dispatches to either `CartesianAcquisitionInfo` or
`NonCartesianAcquisitionInfo`.

- If `trajectory` is `nothing`, constructs `CartesianAcquisitionInfo`
- If `trajectory` is provided, constructs `NonCartesianAcquisitionInfo`

For non-Cartesian acquisitions, `dcf` may be provided as an optional density
compensation array and `subsampling` is not allowed. Leaving `dcf` at `nothing` keeps
the encoding operator's adjoint the true adjoint; see [`NonCartesianAcquisitionInfo`](@ref).

`header` holds the acquisition's metadata (geometry, sequence parameters, anything else): a
[`Header`](@ref), stored as given and not copied, or keywords for one as a `NamedTuple` or another
dictionary. An empty header is created when none is given. Read it with [`header`](@ref) and
attach tags with [`settag!`](@ref). Copies made with `AcquisitionInfo(acq; ...)` and by
preprocessing share the header unless a new one is passed.

`kspace_data` is normally an `AbstractArray` (plain or `NamedDimsArray`). For a Cartesian
acquisition whose frames select *different numbers of samples* it is a [`PartitionedKSpace`](@ref)
instead, one array per frame — a dense array cannot hold a ragged sample axis.

The k-space, the sensitivity maps and `dcf` may be device (GPU) arrays, all of them or none:
`Adapt.adapt(CuArray, acq)` moves an acquisition, and a reconstruction then runs on the device.
The subsampling pattern and the trajectory stay host arrays. See "GPU Reconstruction" in the manual.
"""
function AcquisitionInfo(
        kspace_data = nothing;
        trajectory = nothing,
        is3D::Union{Bool, Nothing} = nothing,
        sensitivity_maps = nothing,
        image_size = nothing,
        subsampling = nothing,
        dcf = nothing,
        shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
        shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
        header = nothing,
    )
    if isnothing(trajectory)
        @argcheck isnothing(dcf) "dcf can only be used with trajectory-based acquisitions"
        return CartesianAcquisitionInfo(
            kspace_data;
            is3D,
            sensitivity_maps,
            image_size,
            subsampling,
            shifted_kspace_dims,
            shifted_image_dims,
            header,
        )
    end
    @argcheck isnothing(subsampling) "subsampling cannot be used with trajectory-based acquisitions"
    return NonCartesianAcquisitionInfo(
        kspace_data;
        trajectory,
        dcf,
        sensitivity_maps,
        image_size,
        shifted_kspace_dims,
        shifted_image_dims,
        header,
    )
end

header(info::AcquisitionInfo) = info.header

# The k-space and the arrays reconstructed against it must all be in host memory or all in device
# memory: every operator built from them runs where the k-space lives.
function _check_same_storage(ksp, x, what::AbstractString)
    (isnothing(ksp) || isnothing(x) || x isa Symbol) && return nothing
    _is_device(ksp) == _is_device(x) && return nothing
    throw(
        ArgumentError(
            "k-space ($(nameof(_array_type_of(ksp)))) and $what ($(nameof(_array_type_of(x)))) must both " *
                "be in host memory or both in device memory; move them together with `Adapt.adapt`."
        )
    )
end

"""
    _copy_with_overrides(config::CartesianAcquisitionInfo; kwargs...)

Type-stable copy helper for `CartesianAcquisitionInfo`: overrides fields with `kwargs`
and constructs a new `CartesianAcquisitionInfo`.
"""
function _copy_with_overrides(config::CartesianAcquisitionInfo; kwargs...)
    kw = kwargs
    kspace_data = get(kw, :kspace_data, config.kspace_data)
    is3D = get(kw, :is3D, config.is3D)
    image_size = get(kw, :image_size, config.image_size)
    sensitivity_maps = get(kw, :sensitivity_maps, config.sensitivity_maps)
    subsampling = get(kw, :subsampling, config.subsampling)
    shifted_kspace_dims = get(kw, :shifted_kspace_dims, config.shifted_kspace_dims)
    shifted_image_dims = get(kw, :shifted_image_dims, config.shifted_image_dims)
    header = get(kw, :header, config.header)
    return CartesianAcquisitionInfo(
        kspace_data;
        is3D,
        image_size,
        sensitivity_maps,
        subsampling,
        shifted_kspace_dims,
        shifted_image_dims,
        header,
    )
end

"""
    _copy_with_overrides(config::NonCartesianAcquisitionInfo; kwargs...)

Type-stable copy helper for `NonCartesianAcquisitionInfo`: overrides non-derived fields with `kwargs`
and constructs a new `NonCartesianAcquisitionInfo` (skipping the derived `:is3D` field).
"""
function _copy_with_overrides(config::NonCartesianAcquisitionInfo; kwargs...)
    kw = kwargs
    kspace_data = get(kw, :kspace_data, config.kspace_data)
    trajectory = get(kw, :trajectory, config.trajectory)
    dcf = get(kw, :dcf, config.dcf)
    sensitivity_maps = get(kw, :sensitivity_maps, config.sensitivity_maps)
    image_size = get(kw, :image_size, config.image_size)
    shifted_kspace_dims = get(kw, :shifted_kspace_dims, config.shifted_kspace_dims)
    shifted_image_dims = get(kw, :shifted_image_dims, config.shifted_image_dims)
    header = get(kw, :header, config.header)
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

CartesianAcquisitionInfo(config::CartesianAcquisitionInfo; kwargs...) =
    _copy_with_overrides(config; kwargs...)

NonCartesianAcquisitionInfo(config::NonCartesianAcquisitionInfo; kwargs...) =
    _copy_with_overrides(config; kwargs...)

AcquisitionInfo(config::CartesianAcquisitionInfo; kwargs...) =
    _copy_with_overrides(config; kwargs...)

AcquisitionInfo(config::NonCartesianAcquisitionInfo; kwargs...) =
    _copy_with_overrides(config; kwargs...)

module RistrettoNIfTIExt

using Ristretto
using Ristretto: ReconImage, header, _spatial_ndims, _export_volume, _lps_affine, _bids_parameters, _json
using NIfTI: NIfTI, NIVolume, niwrite
using FileIO: FileIO, File, @format_str

# LPS to RAS: the first two patient axes point the other way.
const _LPS_TO_RAS = [-1.0 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 1]

function fileio_save(f::File{format"NIfTI"}, img::ReconImage; sidecar::Bool = true)
    path = FileIO.filename(f)
    vol, _, _ = _export_volume(img)
    h = header(img)
    affine = _LPS_TO_RAS * _lps_affine(h, size(vol)[1:3], _spatial_ndims(img))
    voxel_size = Tuple(Float32.(sqrt.(vec(sum(abs2, affine[1:3, 1:3]; dims = 1)))))
    nii = NIVolume(vol; voxel_size, orientation = Matrix{Float32}(affine[1:3, :]), descrip = "Ristretto")
    # Only the sform is meaningful; a qform code with a zero quaternion would claim no rotation.
    nii.header.qform_code = Int16(0)
    niwrite(path, nii)
    if sidecar
        write(_sidecar_path(path), _json(_bids_parameters(h)), "\n")
    end
    return path
end

_sidecar_path(path) = replace(path, r"\.nii(\.gz)?$" => "") * ".json"

function __init__()
    # FileIO's registry has no NIfTI format; `.nii.gz` ends in `.gz`, which FileIO reads as gzip,
    # so that one is saved through `File{format"NIfTI"}`.
    haskey(FileIO.sym2info, :NIfTI) || FileIO.add_format(format"NIfTI", (), ".nii")
    # First in line, so that a saver registered for plain arrays never receives a `ReconImage`.
    savers = FileIO.add_saver(format"NIfTI", @__MODULE__)
    filter!(!=(@__MODULE__), savers)
    pushfirst!(savers, @__MODULE__)
    return nothing
end

end

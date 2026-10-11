module RistrettoDICOMExt

using Ristretto
using Ristretto: ReconImage, header, _export_geometry, _json
using DICOM: DICOM, dcm_write
using FileIO: FileIO, File, @format_str
using LinearAlgebra: cross, dot
using Printf: @sprintf
using Random: RandomDevice

const _MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
const _EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"
# A UID under the 2.25 root (ISO/IEC 9834-8) needs no registered organisation root.
const _IMPLEMENTATION_UID = "2.25.163720398574052957283426938571605934417"

_uid() = "2.25." * string(rand(RandomDevice(), UInt128))

# Decimal (DS) and integer (IS) strings, as the vector of values DICOM.jl writes; a DS value holds
# at most 16 characters.
_ds_value(x::Real) = (s = @sprintf("%.10g", x); length(s) <= 16 ? s : @sprintf("%.6g", x))
_ds(x::Real) = [_ds_value(x)]
_ds(v) = map(_ds_value, collect(v))
_is(x::Integer) = [string(x)]
# Text values have even length, padded with a space.
_even(s::AbstractString) = isodd(ncodeunits(s)) ? s * " " : s

function fileio_save(f::File{format"DCM"}, img::ReconImage; series_description::AbstractString = "Ristretto", series_number::Integer = 1)
    (; vol, extra_names, orientation, spacing, affine) = _export_geometry(img)
    A = affine
    h = header(img)
    nx, ny, nz = size(vol, 1), size(vol, 2), size(vol, 3)
    extra_size = size(vol)[4:end]
    nextra = prod(extra_size; init = 1)
    row_dir, col_dir = orientation[:, 1], orientation[:, 2]
    normal = cross(row_dir, col_dir)

    vol = reshape(vol, nx, ny, nz, nextra)
    peak = maximum(abs, vol; init = 0.0)
    slope = peak > 0 ? Float64(peak) / typemax(UInt16) : 1.0

    study, series, frame = _uid(), _uid(), _uid()
    comments = isempty(h.tags) ? nothing : _even(_json(h.tags))
    path = FileIO.filename(f)
    mkpath(dirname(abspath(path)))
    files = String[]
    for e in 1:nextra, z in 1:nz
        n = length(files) + 1
        position = (A * [0, 0, z - 1, 1])[1:3]
        # DICOM stores rows of the image one after the other, x running along a row.
        pixels = permutedims(round.(UInt16, clamp.(abs.(view(vol, :, :, z, e)) ./ slope, 0, typemax(UInt16))))
        instance = _uid()
        meta = Dict{Tuple{UInt16, UInt16}, Any}(
            (0x0008, 0x0008) => ["DERIVED", "PRIMARY"],
            (0x0008, 0x0016) => _MR_IMAGE_STORAGE,
            (0x0008, 0x0018) => instance,
            (0x0008, 0x0060) => "MR",
            (0x0008, 0x103E) => _even(series_description),
            (0x0010, 0x0010) => "",
            (0x0010, 0x0020) => "",
            (0x0018, 0x0050) => _ds(something(h.slice_thickness, spacing[3])),
            (0x0018, 0x0088) => _ds(spacing[3]),
            (0x0020, 0x000D) => study,
            (0x0020, 0x000E) => series,
            (0x0020, 0x0011) => _is(series_number),
            (0x0020, 0x0013) => _is(n),
            (0x0020, 0x0032) => _ds(position),
            (0x0020, 0x0037) => _ds(vcat(row_dir, col_dir)),
            (0x0020, 0x0052) => frame,
            (0x0020, 0x1041) => _ds(dot(position, normal)),
            (0x0028, 0x0002) => UInt16(1),
            (0x0028, 0x0004) => "MONOCHROME2",
            (0x0028, 0x0010) => UInt16(ny),
            (0x0028, 0x0011) => UInt16(nx),
            (0x0028, 0x0030) => _ds([spacing[2], spacing[1]]),
            (0x0028, 0x0100) => UInt16(16),
            (0x0028, 0x0101) => UInt16(16),
            (0x0028, 0x0102) => UInt16(15),
            (0x0028, 0x0103) => UInt16(0),
            (0x0028, 0x1052) => _ds(0),
            (0x0028, 0x1053) => _ds(slope),
            (0x7FE0, 0x0010) => pixels,
        )
        if nextra > 1
            meta[(0x0020, 0x0100)] = _is(e)
            meta[(0x0020, 0x0105)] = _is(nextra)
        end
        _set_sequence!(meta, h, extra_names, extra_size, CartesianIndices(extra_size)[e])
        isnothing(comments) || (meta[(0x0020, 0x4000)] = comments)
        _set_file_meta!(meta, instance)
        file = nz * nextra == 1 ? path : _numbered(path, n)
        dcm_write(file, DICOM.DICOMData(meta, :little, true, Dict{Tuple{UInt16, UInt16}, String}()))
        push!(files, file)
    end
    return files
end

# `series.dcm` becomes `series_00001.dcm`, `series_00002.dcm`, ...
function _numbered(path, n)
    stem, ext = splitext(path)
    return string(stem, @sprintf("_%05d", n), isempty(ext) ? ".dcm" : ext)
end

function __init__()
    # FileIO lists other savers for DICOM; this one goes first, so that a `ReconImage` reaches it.
    savers = FileIO.add_saver(format"DCM", @__MODULE__)
    filter!(!=(@__MODULE__), savers)
    pushfirst!(savers, @__MODULE__)
    return nothing
end

# The sequence parameters of the image at `index` of the non-image axes `extra_names`.
function _set_sequence!(meta, h, extra_names, extra_size, index)
    params = (
        (0x0018, 0x0080) => h.TR, (0x0018, 0x0081) => h.TE, (0x0018, 0x0082) => h.TI,
        (0x0018, 0x0087) => h.field_strength, (0x0018, 0x1314) => h.flip_angle,
    )
    for (tag, v) in params
        x = _image_value(v, extra_names, extra_size, index)
        isnothing(x) || (meta[tag] = _ds(x))
    end
    return meta
end

# A parameter with one value per contrast (echo, inversion, ...) takes the value of the image at
# `index` on the non-image axis of that length, one named for a contrast if several have it.
# Without such an axis no one value describes the image, and none is written.
const _CONTRAST_AXES = (:contrast, :echo, :inversion)
_image_value(v, extra_names, extra_size, index) = v
function _image_value(v::AbstractVector, extra_names, extra_size, index)
    length(v) == 1 && return only(v)
    candidates = filter(k -> extra_size[k] == length(v), eachindex(extra_size))
    named = filter(k -> extra_names[k] in _CONTRAST_AXES, candidates)
    k = !isempty(named) ? first(named) : length(candidates) == 1 ? only(candidates) : nothing
    return isnothing(k) ? nothing : v[index[k]]
end

# The file meta group (0002), whose group length is the byte count of the elements after it.
function _set_file_meta!(meta, instance)
    group = Dict{Tuple{UInt16, UInt16}, Any}(
        (0x0002, 0x0001) => UInt8[0x00, 0x01],
        (0x0002, 0x0002) => _MR_IMAGE_STORAGE,
        (0x0002, 0x0003) => instance,
        (0x0002, 0x0010) => _EXPLICIT_VR_LITTLE_ENDIAN,
        (0x0002, 0x0012) => _IMPLEMENTATION_UID,
    )
    io = IOBuffer()
    dcm_write(io, DICOM.DICOMData(group, :little, true, Dict{Tuple{UInt16, UInt16}, String}()); preamble = false)
    meta[(0x0002, 0x0000)] = UInt32(position(io))
    merge!(meta, group)
    return meta
end

end

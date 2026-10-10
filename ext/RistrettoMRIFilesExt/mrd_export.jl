# ISMRMRD image data types and image types.
_mrd_data_type(::Type{Float32}) = 5
_mrd_data_type(::Type{Float64}) = 6
_mrd_data_type(::Type{ComplexF32}) = 7
_mrd_data_type(::Type{ComplexF64}) = 8
_mrd_eltype(::Type{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}} = T
_mrd_eltype(::Type{<:Complex}) = ComplexF32
_mrd_eltype(::Type) = Float32

function FileIO.save(f::ISMRMRDFile, img::ReconImage; group::AbstractString = "image_0")
    path = f.filename
    vol, extra_names, multislice = _export_volume(img)
    h = header(img)
    nd = _spatial_ndims(img)
    nx, ny, nz = size(vol, 1), size(vol, 2), size(vol, 3)
    extra_size = size(vol)[4:end]
    nextra = prod(extra_size; init = 1)
    A = _lps_affine(h, (nx, ny, nz), nd)
    spacing = [norm(A[1:3, j]) for j in 1:3]
    dirs = [A[1:3, j] ./ spacing[j] for j in 1:3]
    T = _mrd_eltype(eltype(vol))
    data = reshape(convert(Array{T}, vol), nx, ny, nz, nextra)

    # A 3D image is one MRD image; each slice of a 2D image is one.
    per_image_z = nd == 3 ? nz : 1
    nslices = nd == 3 ? 1 : nz
    images = Array{T}(undef, nx, ny, per_image_z, 1, nslices * nextra)
    headers = Vector{NTuple{_IMAGE_HEADER_SIZE, UInt8}}(undef, nslices * nextra)
    attributes = Vector{String}(undef, nslices * nextra)
    time_axis = findfirst(==(:time), extra_names)
    meta = _meta_xml(h.tags)
    # The field of view of one image: a slice of a 2D image is as thick as its excitation.
    fov = Float32.((spacing[1] * nx, spacing[2] * ny, nd == 3 ? spacing[3] * nz : something(h.slice_thickness, spacing[3])))
    n = 0
    for e in 1:nextra, s in 1:nslices
        n += 1
        zs = nd == 3 ? (1:nz) : (s:s)
        images[:, :, :, 1, n] = data[:, :, zs, e]
        # MRD positions an image by its centre.
        centre = A * [nx ÷ 2, ny ÷ 2, nd == 3 ? nz ÷ 2 : s - 1, 1]
        phase = isnothing(time_axis) ? 0 : CartesianIndices(extra_size)[e][time_axis] - 1
        headers[n] = _image_header(;
            data_type = _mrd_data_type(T), matrix_size = (nx, ny, per_image_z), fov,
            position = centre[1:3], dirs, slice = nd == 3 ? 0 : s - 1, phase,
            image_index = n, attribute_string_len = ncodeunits(meta),
        )
        attributes[n] = meta
    end

    HDF5.h5open(path, "w") do file
        write(file, "/dataset/xml", [_xml_header(h, (nx, ny, per_image_z), fov, nslices)])
        g = HDF5.create_group(file["dataset"], group)
        _write_headers(g, "header", headers)
        _write_data(g, "data", images)
        write(g, "attributes", attributes)
    end
    return path
end

# ---------------------------------------------------------------- ISMRMRD image header

const _IMAGE_HEADER_SIZE = 198

# (name, element type, count) of every ImageHeader field, in file order.
const _IMAGE_HEADER_FIELDS = (
    ("version", UInt16, 1), ("data_type", UInt16, 1), ("flags", UInt64, 1),
    ("measurement_uid", UInt32, 1), ("matrix_size", UInt16, 3), ("field_of_view", Float32, 3),
    ("channels", UInt16, 1), ("position", Float32, 3), ("read_dir", Float32, 3),
    ("phase_dir", Float32, 3), ("slice_dir", Float32, 3), ("patient_table_position", Float32, 3),
    ("average", UInt16, 1), ("slice", UInt16, 1), ("contrast", UInt16, 1), ("phase", UInt16, 1),
    ("repetition", UInt16, 1), ("set", UInt16, 1), ("acquisition_time_stamp", UInt32, 1),
    ("physiology_time_stamp", UInt32, 3), ("image_type", UInt16, 1), ("image_index", UInt16, 1),
    ("image_series_index", UInt16, 1), ("user_int", Int32, 8), ("user_float", Float32, 8),
    ("attribute_string_len", UInt32, 1),
)

function _image_header(; data_type, matrix_size, fov, position, dirs, slice, phase, image_index, attribute_string_len)
    values = Dict{String, Any}(
        "version" => 1, "data_type" => data_type, "matrix_size" => matrix_size,
        "field_of_view" => fov, "channels" => 1, "position" => position,
        "read_dir" => dirs[1], "phase_dir" => dirs[2], "slice_dir" => dirs[3],
        "slice" => slice, "phase" => phase,
        "image_type" => data_type in (7, 8) ? 5 : 1, "image_index" => image_index,
        "attribute_string_len" => attribute_string_len,
    )
    io = IOBuffer()
    for (name, T, count) in _IMAGE_HEADER_FIELDS
        v = get(values, name, count == 1 ? zero(T) : zeros(T, count))
        for x in (count == 1 ? (v,) : v)
            write(io, htol(convert(T, x)))
        end
    end
    bytes = take!(io)
    @assert length(bytes) == _IMAGE_HEADER_SIZE
    return NTuple{_IMAGE_HEADER_SIZE, UInt8}(bytes)
end

function _image_header_type()
    t = API.h5t_create(API.H5T_COMPOUND, _IMAGE_HEADER_SIZE)
    offset = 0
    for (name, T, count) in _IMAGE_HEADER_FIELDS
        if count == 1
            API.h5t_insert(t, name, offset, HDF5.hdf5_type_id(T))
        else
            a = API.h5t_array_create(HDF5.hdf5_type_id(T), Cuint(1), API.hsize_t[count])
            API.h5t_insert(t, name, offset, a)
            API.h5t_close(a)
        end
        offset += sizeof(T) * count
    end
    return t
end

function _write_headers(file, name, headers)
    t = _image_header_type()
    space = API.h5s_create_simple(1, API.hsize_t[length(headers)], API.hsize_t[length(headers)])
    dset = API.h5d_create(file, name, t, space, API.H5P_DEFAULT, API.H5P_DEFAULT, API.H5P_DEFAULT)
    API.h5d_write(dset, t, API.H5S_ALL, API.H5S_ALL, API.H5P_DEFAULT, headers)
    API.h5d_close(dset)
    API.h5s_close(space)
    API.h5t_close(t)
    return nothing
end

# ISMRMRD stores complex samples as a compound of `real` and `imag`.
_write_data(file, name, data::Array{<:Real}) = (write(file, name, data); nothing)
function _write_data(file, name, data::Array{Complex{T}}) where {T}
    t = MRIFiles.get_hdf5type_complex(T)
    dims = API.hsize_t[reverse(size(data))...]
    space = API.h5s_create_simple(ndims(data), dims, dims)
    dset = API.h5d_create(file, name, t, space, API.H5P_DEFAULT, API.H5P_DEFAULT, API.H5P_DEFAULT)
    API.h5d_write(dset, t, API.H5S_ALL, API.H5S_ALL, API.H5P_DEFAULT, data)
    API.h5d_close(dset)
    API.h5s_close(space)
    API.h5t_close(t)
    return nothing
end

# ---------------------------------------------------------------- XML

function _xml_header(h, matrix, fov, nslices)
    params = Dict{String, Any}(
        "encodedSize" => collect(Int, matrix), "encodedFOV" => collect(Float64, fov),
        "reconSize" => collect(Int, matrix), "reconFOV" => collect(Float64, fov),
        "trajectory" => "other",
        "enc_lim_slice" => MRIFiles.Limit(0, nslices - 1, 0),
        "H1resonanceFrequency_Hz" => isnothing(h.field_strength) ? 0 : round(Int, h.field_strength * _GAMMA_HZ_PER_T),
    )
    isnothing(h.field_strength) || (params["systemFieldStrength_T"] = h.field_strength)
    isnothing(h.TR) || (params["TR"] = h.TR)
    isnothing(h.TE) || (params["TE"] = h.TE)
    isnothing(h.TI) || (params["TI"] = h.TI)
    isnothing(h.flip_angle) || (params["flipAngle_deg"] = h.flip_angle)
    return MRIFiles.GeneralParametersToXML(params)
end

# Tags as an ISMRMRD meta container: one `meta` element per tag.
function _meta_xml(tags)
    io = IOBuffer()
    print(io, "<?xml version=\"1.0\"?><ismrmrdMeta>")
    for (k, v) in sort!(collect(tags); by = first)
        print(io, "<meta><name>", _escape(k), "</name>")
        for x in (v isa AbstractVector ? v : (v,))
            print(io, "<value>", _escape(string(x)), "</value>")
        end
        print(io, "</meta>")
    end
    print(io, "</ismrmrdMeta>")
    return String(take!(io))
end

_escape(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

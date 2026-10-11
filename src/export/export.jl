# What the writers in the NIfTI, DICOM and MRIFiles extensions share: the layout of the volume
# and its voxel-to-patient affine, and the header as text.

# ---------------------------------------------------------------- layout and geometry

# `img` as the writers need it:
# - `vol`: its data as a host `Array` whose first three axes are x, y and z (a 2D image without a
#   slice axis gets one of length one), not copied when it already is one;
# - `extra_names`: the names of its other axes;
# - `nd`: the number of image axes, 2 or 3;
# - `orientation` (columns: the LPS directions of x, y, z), `spacing` (mm, along x, y, z) and
#   `affine`, the 4×4 map from 0-based voxel indices to LPS coordinates in mm.
# What the header lacks is filled in so that a file can still be written: identity orientation,
# 1 mm voxels, a slice spacing equal to the slice thickness, and the image centre at the origin.
function _export_geometry(img::ReconImage)
    data = unname(parent(img))
    data isa Array || (data = Array(data))
    nd = _spatial_ndims(img)
    @argcheck nd in (2, 3) "only 2D and 3D images can be exported, this one has $nd image axes"
    names = parent(img) isa NamedDimsArray ? collect(Symbol, dimnames(img)) : fill(:_, ndims(data))
    vol, extra_names = if nd == 3 || (ndims(data) >= 3 && names[3] in (:z, :slice))
        data, names[4:end]
    else
        reshape(data, size(data, 1), size(data, 2), 1, size(data)[3:end]...), names[3:end]
    end
    h = header(img)
    vol_size = size(vol)[1:3]
    M = something(h.orientation, Matrix{Float64}(I, 3, 3)) * Diagonal(collect(_export_spacing(h, vol_size, nd)))
    offset = isnothing(h.offset) ? -(M * collect(Float64, vol_size .÷ 2)) : collect(h.offset)
    # The header's slice spacing is signed along the slice normal, and its orientation need not
    # have unit columns; the writers want unit directions and positive spacings.
    spacing = Tuple(norm(c) for c in eachcol(M))
    orientation = M ./ collect(spacing)'
    return (; vol, extra_names, nd, orientation, spacing, affine = [M offset; 0 0 0 1])
end

function _export_spacing(h::Header, vol_size, nd)
    sp = h.spacing
    if isnothing(sp) || length(sp) != nd
        fov = h.fov
        sp = !isnothing(fov) && length(fov) == nd ? fov ./ vol_size[1:nd] : nothing
    end
    if isnothing(sp)
        @warn "the header has no spacing or fov; exporting with 1 mm voxels" maxlog = 1
        sp = ntuple(_ -> 1.0, nd)
    end
    nd == 3 && return Tuple(Float64.(sp))
    return (Float64(sp[1]), Float64(sp[2]), something(h.slice_spacing, h.slice_thickness, 1.0))
end

# ---------------------------------------------------------------- metadata as text

# Sequence parameters under their BIDS sidecar names, in seconds where BIDS uses seconds.
function _bids_parameters(h::Header)
    d = Dict{String, Any}()
    seconds(ms) = ms .* 1.0e-3
    TE, TR, TI, flip, B0, thickness = h.TE, h.TR, h.TI, h.flip_angle, h.field_strength, h.slice_thickness
    isnothing(TE) || (d["EchoTime"] = seconds(TE))
    isnothing(TR) || (d["RepetitionTime"] = seconds(TR))
    isnothing(TI) || (d["InversionTime"] = seconds(TI))
    isnothing(flip) || (d["FlipAngle"] = flip)
    isnothing(B0) || (d["MagneticFieldStrength"] = B0)
    isnothing(thickness) || (d["SliceThickness"] = thickness)
    for (k, v) in h.extra
        d[string(k)] = v
    end
    isempty(h.tags) || (d["Tags"] = h.tags)
    return d
end

# A minimal JSON writer for the values a header holds; anything else is written as its string.
_json(x) = sprint(_json, x)
_json(io::IO, x::AbstractString) = (print(io, '"'); _json_escape(io, x); print(io, '"'))
_json(io::IO, x::Symbol) = _json(io, string(x))
_json(io::IO, x::Bool) = print(io, x)
_json(io::IO, x::Integer) = print(io, x)
_json(io::IO, x::AbstractFloat) = isfinite(x) ? print(io, x) : print(io, "null")
_json(io::IO, ::Nothing) = print(io, "null")
_json(io::IO, x::Union{AbstractVector, Tuple}) =
    (print(io, '['); join(io, (_json(v) for v in x), ", "); print(io, ']'))
function _json(io::IO, x::AbstractDict)
    print(io, '{')
    join(io, (_json(string(k)) * ": " * _json(v) for (k, v) in sort!(collect(x); by = p -> string(first(p)))), ", ")
    return print(io, '}')
end
_json(io::IO, x) = _json(io, string(x))

function _json_escape(io::IO, s::AbstractString)
    for c in String(s)::String
        if c == '"' || c == '\\'
            print(io, '\\', c)
        elseif c < ' '
            print(io, "\\u", string(UInt16(c); base = 16, pad = 4))
        else
            print(io, c)
        end
    end
    return nothing
end

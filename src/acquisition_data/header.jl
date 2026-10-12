# The metadata header carried by an `AcquisitionInfo` and by the `ReconImage` reconstructed from it.

"""
    Header(; kwargs...) <: AbstractDict{Symbol, Any}
    Header(dict_or_pairs)

Metadata of an acquisition or an image, built from any keyword arguments. Every entry is optional.

The keys Ristretto knows are fields of the header, checked and converted when set; any other key
is kept as given in `extra`. Lengths are in mm, times in ms, angles in degrees, the field in T, and
positions in the patient coordinate system LPS (x towards the patient's left, y posterior, z
superior), as MRD and DICOM store them.

| key | meaning |
|---|---|
| `fov` | per encoded image axis, 2 or 3 entries |
| `spacing` | centre-to-centre voxel distance along x, y and z, positive; a 2D image's third entry, when given, is the distance between neighbouring slices |
| `voxel_size` | extent of a voxel along each encoded axis, when it is not `spacing` (a strided selection keeps it); unset means equal to `spacing` |
| `slice_thickness` | excited slice thickness of a 2D image |
| `orientation` | 3×3, columns the unit directions of the axes x, y and z (read, phase, slice) |
| `offset` | centre of the first voxel, index `(1, 1, 1)`; voxel `i` is at `offset + orientation * ((i .- 1) .* spacing)` |
| `TE`, `TR`, `TI`, `flip_angle`, `field_strength` | sequence parameters |
| `tags` | user tags, see [`settag!`](@ref) |

A header is a dictionary over the keys that are set: `h[:TE]` and `h[:protocol]` throw a
`KeyError` when absent, and `get`, `haskey`, `keys` and iteration see only set keys. A known key
is also a property, `nothing` when unset (`h.fov`); an unknown property throws.

```julia
h = Header(; fov = (240, 180), TE = 4.2, protocol = "t1_se")
h.fov            # (240.0, 180.0)
h[:protocol]     # "t1_se"
h.offset         # nothing
```
"""
mutable struct Header <: AbstractDict{Symbol, Any}
    fov::Union{Nothing, NTuple{2, Float64}, NTuple{3, Float64}}
    spacing::Union{Nothing, NTuple{2, Float64}, NTuple{3, Float64}}
    voxel_size::Union{Nothing, NTuple{2, Float64}, NTuple{3, Float64}}
    slice_thickness::Union{Nothing, Float64}
    orientation::Union{Nothing, Matrix{Float64}}
    offset::Union{Nothing, NTuple{3, Float64}}
    TE::Union{Nothing, Float64, Vector{Float64}}
    TR::Union{Nothing, Float64}
    TI::Union{Nothing, Float64, Vector{Float64}}
    flip_angle::Union{Nothing, Float64, Vector{Float64}}
    field_strength::Union{Nothing, Float64}
    tags::Dict{String, Any}
    extra::Dict{Symbol, Any}

    function Header(pairs)
        h = new(
            nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, nothing, nothing,
            Dict{String, Any}(), Dict{Symbol, Any}(),
        )
        for (k, v) in pairs
            h[Symbol(k)] = v
        end
        return h
    end
end
Header(; kwargs...) = Header(kwargs)
Header(nt::NamedTuple) = Header(pairs(nt))

const _HEADER_KNOWN_KEYS = (
    :fov, :spacing, :voxel_size, :slice_thickness, :orientation, :offset,
    :TE, :TR, :TI, :flip_angle, :field_strength,
)

_is_known(k::Symbol) = k in _HEADER_KNOWN_KEYS

# ---------------------------------------------------------------- properties

function Base.setproperty!(h::Header, k::Symbol, v)
    k === :extra && return setfield!(h, :extra, Dict{Symbol, Any}(Symbol(a) => b for (a, b) in v))
    hasfield(Header, k) || throw(ArgumentError("Header has no property `$k`; set other keys with `h[:$k] = v`"))
    return setfield!(h, k, _normalize(Val(k), v))
end

_normalize(::Val, v) = v
_normalize(::Val{:fov}, v) = _axis_tuple(:fov, v)
function _normalize(::Val{:spacing}, v)
    sp = _axis_tuple(:spacing, v)
    @argcheck isnothing(sp) || all(>(0), sp) "spacing must be positive; a reversed axis is a reversed orientation column"
    return sp
end
function _normalize(::Val{:voxel_size}, v)
    vs = _axis_tuple(:voxel_size, v)
    @argcheck isnothing(vs) || all(>(0), vs) "voxel_size must be positive"
    return vs
end
_normalize(::Val{:slice_thickness}, v) = _float(v)
_normalize(::Val{:TR}, v) = _float(v)
_normalize(::Val{:field_strength}, v) = _float(v)
_normalize(::Val{:TE}, v) = _float_or_vector(v)
_normalize(::Val{:TI}, v) = _float_or_vector(v)
_normalize(::Val{:flip_angle}, v) = _float_or_vector(v)
function _normalize(::Val{:offset}, v)
    isnothing(v) && return nothing
    @argcheck length(v) == 3 "offset must have 3 entries, got $(length(v))"
    return Tuple(Float64.(v))::NTuple{3, Float64}
end
function _normalize(::Val{:orientation}, m)
    isnothing(m) && return nothing
    @argcheck size(m) == (3, 3) "orientation must be a 3×3 matrix whose columns are the image axes' directions"
    return Matrix{Float64}(m)
end
_normalize(::Val{:tags}, v) = Dict{String, Any}(string(k) => x for (k, x) in v)

function _axis_tuple(k, v)
    isnothing(v) && return nothing
    @argcheck length(v) in (2, 3) "$k must have 2 or 3 entries, one per image axis, got $(length(v))"
    return Tuple(Float64.(v))
end
_float(::Nothing) = nothing
_float(x::Real) = Float64(x)
_float_or_vector(::Nothing) = nothing
_float_or_vector(x::Real) = Float64(x)
_float_or_vector(x::AbstractVector) = Vector{Float64}(x)

# ---------------------------------------------------------------- dictionary interface

_isset(h::Header, k::Symbol) =
    _is_known(k) ? !isnothing(getfield(h, k)) : k === :tags ? !isempty(h.tags) : haskey(h.extra, k)

function Base.haskey(h::Header, k::Symbol)
    return _isset(h, k)
end
Base.get(h::Header, k::Symbol, default) = _isset(h, k) ? _getkey(h, k) : default
Base.get(f::Base.Callable, h::Header, k::Symbol) = _isset(h, k) ? _getkey(h, k) : f()
Base.getindex(h::Header, k::Symbol) = _isset(h, k) ? _getkey(h, k) : throw(KeyError(k))
_getkey(h::Header, k::Symbol) = (_is_known(k) || k === :tags) ? getfield(h, k) : h.extra[k]

function Base.setindex!(h::Header, v, k::Symbol)
    if _is_known(k) || k === :tags
        setproperty!(h, k, v)
    else
        h.extra[k] = v
    end
    return h
end

function Base.delete!(h::Header, k::Symbol)
    if _is_known(k)
        setfield!(h, k, nothing)
    elseif k === :tags
        empty!(h.tags)
    else
        delete!(h.extra, k)
    end
    return h
end

function _set_keys(h::Header)
    ks = Symbol[k for k in _HEADER_KNOWN_KEYS if !isnothing(getfield(h, k))]
    isempty(h.tags) || push!(ks, :tags)
    append!(ks, keys(h.extra))
    return ks
end

Base.keys(h::Header) = _set_keys(h)
Base.length(h::Header) = length(_set_keys(h))
function Base.iterate(h::Header, state = (_set_keys(h), 1))
    ks, i = state
    i > length(ks) && return nothing
    return (ks[i] => _getkey(h, ks[i]), (ks, i + 1))
end

"""
    copy(h::Header)

A header sharing no mutable state with `h`, so tagging the copy leaves `h` alone.
"""
Base.copy(h::Header) = deepcopy(h)

function Base.show(io::IO, h::Header)
    print(io, "Header(")
    join(io, ("$k = $(repr(v; context = io))" for (k, v) in h), ", ")
    return print(io, ")")
end
Base.show(io::IO, ::MIME"text/plain", h::Header) = show(io, h)

# ---------------------------------------------------------------- geometry

# The name of the slice axis of a multi-slice 2D image, the axis right after `:x` and `:y`.
const _SLICE_AXIS = :z

# The voxel at the centre of an axis of `n` voxels, 0-based: `n ÷ 2`, the FFT-shifted origin, which
# is where MRD positions a slice or slab. `M` maps 0-based voxel steps to mm (orientation × spacing).
_centre_from_offset(offset, M, n) = collect(Float64, offset) .+ M * collect(Float64, n .÷ 2)
_offset_from_centre(centre, M, n) = collect(Float64, centre) .- M * collect(Float64, n .÷ 2)

# Fill `spacing` from `fov` and the image size `img_size` (one entry per encoded axis) when only
# `fov` is known, and warn when `fov`, `spacing` and the image size disagree. A scanner's `fov` may
# describe the oversampled grid, so a mismatch is not an error.
function _complete_geometry!(h::Header, img_size)
    isnothing(img_size) && return h
    nd = length(img_size)
    fov, sp, vs = h.fov, h.spacing, h.voxel_size
    for (k, v) in ((:fov, fov), (:voxel_size, vs))
        if !isnothing(v) && length(v) != nd
            @warn "the header's $k has $(length(v)) entries but the image has $nd axes" maxlog = 1
            return h
        end
    end
    if !isnothing(sp) && !(length(sp) == nd || (nd == 2 && length(sp) == 3))
        @warn "the header's spacing has $(length(sp)) entries but the image has $nd axes" maxlog = 1
        return h
    end
    isnothing(fov) && return h
    if isnothing(sp)
        h.spacing = fov ./ Tuple(img_size)
    elseif !all(i -> isapprox(fov[i], sp[i] * img_size[i]; rtol = 1.0e-3), 1:nd)
        @warn "the header's fov $fov does not equal spacing × image size $(sp[1:nd] .* Tuple(img_size))" maxlog = 1
    end
    return h
end

# A header from what a constructor was given: an empty one for `nothing`, a `Header` as given, a
# `NamedTuple` or another dictionary converted.
_to_header(::Nothing) = Header()
_to_header(h::Header) = h
_to_header(h) = Header(pairs(h))

"""
    header(x) -> Header

The metadata [`Header`](@ref) of an [`AcquisitionInfo`](@ref) or a [`ReconImage`](@ref); a
`Header` is its own header, so the tag functions also take one.
"""
function header end
header(h::Header) = h

"""
    settag!(x, key, value) -> x

Attach a tag to an [`AcquisitionInfo`](@ref) or a [`ReconImage`](@ref). Tags live in the header,
travel from an acquisition to the images reconstructed from it, and are written by the export
functions. An acquisition stores the header it is given, so tagging a `Header` object also tags
every acquisition holding it; a `ReconImage` holds its own copy, so tagging an image leaves its
acquisition unchanged.
"""
settag!(x, key, value) = (header(x).tags[string(key)] = value; x)

"""
    gettag(x, key[, default])

The tag `key` of `x` (see [`settag!`](@ref)), or `default` when it is not set; without a
`default`, a missing tag throws a `KeyError`.
"""
gettag(x, key) = header(x).tags[string(key)]
gettag(x, key, default) = get(header(x).tags, string(key), default)

"""
    tags(x) -> Dict{String, Any}

All tags of `x` (see [`settag!`](@ref)).
"""
tags(x) = header(x).tags

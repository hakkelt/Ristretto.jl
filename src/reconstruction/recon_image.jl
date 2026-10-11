"""
    ReconImage(data, header = Header(); components = nothing, spatial_ndims) <: AbstractArray

The result of [`reconstruct`](@ref): the image `data` (a `NamedDimsArray` when the k-space was
named, on the host or a device) together with a copy of the acquisition's [`header`](@ref), which
carries the geometry (`fov`, `spacing`, `orientation`, `offset`), the sequence
parameters and the tags (see [`Header`](@ref)). The first `spatial_ndims` axes of `data` are the
image axes. When it is not given, it is the length of the header's `fov`, or else the number of
leading axes named `:x`, `:y`, `:z`, or else the length of its `spacing`; a multi-slice 2D image
whose header has no `fov` therefore needs it. A `fov` without a `spacing` gives the spacing.

It is an `AbstractArray` that behaves as `data`: positional indexing, `size` (also by dimension
name, `size(img, :x)`), broadcasting and `Array(img)` all act on the image. `parent(img)` returns `data` and `dimnames(img)` its dimension
names. Keyword indexing names the dimensions, as for a `NamedDimsArray`, and keeps the header with
its geometry updated to the part selected:

```julia
img[z = 5]            # one slice: offset moved to it, the slice axis dropped from the geometry
img[x = 33:96]        # a crop: offset moved to x = 33, fov shrinks
view(img; time = 1)   # non-spatial axes leave the geometry alone
```

A reconstruction with [`Component`](@ref)s returns the sum of the components as `data` and keeps
the components themselves, as arrays shaped like `data` and described by the image's header:
`img.lowrank` is the component named `lowrank`, `components(img)` all of them as a `NamedTuple`.
Keyword indexing selects the same part of every component; [`drop_components`](@ref) releases
them.
"""
struct ReconImage{T, N, A <: AbstractArray{T, N}, C <: Union{Nothing, NamedTuple}} <: AbstractArray{T, N}
    data::A
    header::Header
    components::C
    spatial_ndims::Int
    function ReconImage(data::A, header::Header, components::C, spatial_ndims::Integer) where {T, N, A <: AbstractArray{T, N}, C}
        @argcheck 0 <= spatial_ndims <= N "spatial_ndims must be between 0 and $N"
        if !isnothing(components)
            collisions = filter(in(_RECON_IMAGE_RESERVED_NAMES), keys(components))
            @argcheck isempty(collisions) "Component name(s) $collisions collide with ReconImage's own field(s) $_RECON_IMAGE_RESERVED_NAMES; rename the component(s)."
        end
        return new{T, N, A, C}(data, header, components, spatial_ndims)
    end
end
function ReconImage(data::AbstractArray, header = Header(); components = nothing, spatial_ndims::Union{Nothing, Integer} = nothing)
    h = copy(_to_header(header))
    nd = something(spatial_ndims, _guess_spatial_ndims(data, h))
    nd <= ndims(data) && _complete_geometry!(h, size(data)[1:nd])
    return ReconImage(data, h, components, nd)
end

# `ReconImage`'s real properties; a component of one of these names would be unreachable through
# `img.<name>`, so `check_components` rejects it up front.
const _RECON_IMAGE_RESERVED_NAMES = (:data, :header, :components)

function _guess_spatial_ndims(data, h::Header)
    fov = h.fov
    isnothing(fov) || return min(length(fov), ndims(data))
    if data isa NamedDimsArray
        names = dimnames(data)
        n = 0
        while n < min(3, length(names)) && names[n + 1] === (:x, :y, :z)[n + 1]
            n += 1
        end
        return n
    end
    sp = h.spacing
    return min(isnothing(sp) ? 2 : length(sp), ndims(data))
end

header(img::ReconImage) = getfield(img, :header)
_spatial_ndims(img::ReconImage) = getfield(img, :spatial_ndims)
Base.parent(img::ReconImage) = getfield(img, :data)

Base.size(img::ReconImage) = size(parent(img))
Base.axes(img::ReconImage) = axes(parent(img))
Base.size(img::ReconImage, d::Symbol) = size(parent(img), d)
Base.axes(img::ReconImage, d::Symbol) = axes(parent(img), d)
Base.IndexStyle(::Type{<:ReconImage{T, N, A}}) where {T, N, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(img::ReconImage, i::Int...) = parent(img)[i...]
Base.@propagate_inbounds Base.setindex!(img::ReconImage, v, i::Int...) = (parent(img)[i...] = v; img)
Base.similar(img::ReconImage, ::Type{S}, dims::Dims) where {S} = similar(unname(parent(img)), S, dims)
Base.copy(img::ReconImage) = ReconImage(copy(parent(img)), copy(header(img)), _map_components(copy, img), _spatial_ndims(img))
Base.Array(img::ReconImage) = Array(unname(parent(img)))
Base.convert(::Type{Array}, img::ReconImage) = Array(img)
Base.Broadcast.broadcastable(img::ReconImage) = parent(img)
# Reductions run on the image itself, so that a device image is reduced on the device.
Base.mapreduce(f, op, img::ReconImage; kwargs...) = mapreduce(f, op, parent(img); kwargs...)
LinearAlgebra.norm(img::ReconImage, p::Real = 2) = norm(parent(img), p)
LinearAlgebra.dot(a::ReconImage, b::ReconImage) = dot(parent(a), parent(b))

NamedDims.unname(img::ReconImage) = unname(parent(img))
NamedDims.dimnames(img::ReconImage) = dimnames(parent(img))
NamedDims.dimnames(img::ReconImage, d::Integer) = dimnames(parent(img), d)
NamedDims.NamedDimsArray(img::ReconImage) =
    parent(img) isa NamedDimsArray ? parent(img) : throw(ArgumentError("this ReconImage has no dimension names"))

Adapt.adapt_structure(to, img::ReconImage) =
    ReconImage(_adapt_any(to, parent(img)), copy(header(img)), _map_components(c -> Adapt.adapt(to, c), img), _spatial_ndims(img))
_is_device(img::ReconImage) = _is_device(parent(img))
_storage_template(img::ReconImage) = _storage_template(parent(img))

_map_components(f, img::ReconImage) = (c = getfield(img, :components); isnothing(c) ? nothing : map(f, c))

"""
    components(img::ReconImage) -> NamedTuple

The components of a reconstruction with [`Component`](@ref)s, by name.
"""
function components(img::ReconImage)
    c = getfield(img, :components)
    isnothing(c) && throw(ArgumentError("this ReconImage holds no components; it was not reconstructed with `Component`s"))
    return c
end

"""
    drop_components(img::ReconImage) -> ReconImage

`img` without its components: the same image data and a copy of its header, so that the
components can be garbage-collected once nothing else refers to them.
"""
drop_components(img::ReconImage) = ReconImage(parent(img), copy(header(img)), nothing, _spatial_ndims(img))

"""
    total_image(img::ReconImage)

The image itself, without the header: the sum of the components for a reconstruction with
[`Component`](@ref)s.
"""
total_image(img::ReconImage) = parent(img)

function Base.getproperty(img::ReconImage, name::Symbol)
    name in _RECON_IMAGE_RESERVED_NAMES && return getfield(img, name)
    c = getfield(img, :components)
    !isnothing(c) && haskey(c, name) && return c[name]
    throw(ArgumentError("ReconImage has no property `$name`; available properties: $(join(propertynames(img), ", "))."))
end

function Base.propertynames(img::ReconImage, ::Bool = false)
    c = getfield(img, :components)
    return isnothing(c) ? _RECON_IMAGE_RESERVED_NAMES : (_RECON_IMAGE_RESERVED_NAMES..., keys(c)...)
end

function Base.show(io::IO, ::MIME"text/plain", img::ReconImage)
    print(io, "ReconImage{", eltype(img), "} of size ", join(size(img), "×"))
    parent(img) isa NamedDimsArray && print(io, " ", dimnames(img))
    c = getfield(img, :components)
    isnothing(c) || print(io, " with components ", join(keys(c), ", "))
    sp = header(img).spacing
    isnothing(sp) || print(io, ", spacing ", join(sp, "×"), " mm")
    return nothing
end

# ---------------------------------------------------------------- keyword indexing

Base.getindex(img::ReconImage; kwargs...) = _keyword_index(getindex, img, values(kwargs))
Base.view(img::ReconImage; kwargs...) = _keyword_index(view, img, values(kwargs))

function _keyword_index(f, img::ReconImage, sel::NamedTuple)
    data = parent(img)
    data isa NamedDimsArray || throw(ArgumentError("keyword indexing needs a ReconImage with dimension names"))
    out = f(data; pairs(sel)...)
    out isa AbstractArray || return out
    comps = _map_components(c -> f(c; pairs(sel)...), img)
    h, nd = _slice_geometry(header(img), dimnames(data), _spatial_ndims(img), sel)
    return ReconImage(out, h, comps, nd)
end

# The header of a part of an image. Its geometry has three axes: the image axes, and for a 2D
# image the slice axis (`_SLICE_AXIS`, right after `:x` and `:y`), whose spacing is `spacing[3]`.
# For each axis selected, the offset moves to the first voxel kept and the axis shrinks to the
# voxels kept (its spacing scaled by the step of a range, reversed for a negative step), or is
# dropped when an integer selects one, which makes its voxel size the slice thickness. A stride
# leaves the voxel size alone, so `voxel_size` is set once it differs from the spacing. A selection
# with no constant step has no geometry the header can describe, so it drops the geometry with a
# warning. Any other axis leaves the geometry alone. Returns the header and the number of image
# axes kept.
function _slice_geometry(h0::Header, names, nd::Int, sel::NamedTuple)
    h = copy(h0)
    geometry_names = nd == 3 ? names[1:3] : (names[1:2]..., length(names) >= 3 && names[3] === _SLICE_AXIS ? names[3] : nothing)
    sp0, fov0, vs0, orientation0 = h.spacing, h.fov, h.voxel_size, h.orientation
    spacing = Union{Nothing, Float64}[isnothing(sp0) || g > length(sp0) ? nothing : sp0[g] for g in 1:3]
    fov = Union{Nothing, Float64}[isnothing(fov0) || g > length(fov0) ? nothing : fov0[g] for g in 1:3]
    voxel_size = isnothing(vs0) || length(vs0) != nd ? spacing[1:nd] : Union{Nothing, Float64}[vs0...]
    offset = isnothing(h.offset) ? nothing : collect(h.offset)
    orientation = isnothing(orientation0) ? nothing : copy(orientation0)
    dropped = falses(3)
    for (name, idx0) in pairs(sel)
        g = findfirst(==(name), geometry_names)
        idx = _regular_index(idx0)
        (isnothing(g) || idx isa Colon) && continue
        if isnothing(idx)
            @warn "selecting voxels with no constant step along `$name` drops the image geometry" maxlog = 1
            offset = nothing
            fill!(spacing, nothing)
            fill!(fov, nothing)
            continue
        end
        offset = _moved(offset, orientation, g, idx, spacing[g])
        if idx isa Integer
            dropped[g] = true
            g <= nd && !isnothing(voxel_size[g]) && (h.slice_thickness = voxel_size[g])
        else
            isnothing(spacing[g]) || (spacing[g] *= abs(step(idx)))
            fov[g] = isnothing(spacing[g]) ? nothing : length(idx) * spacing[g]
            step(idx) < 0 && (orientation = _flipped(orientation, g))
        end
    end
    image_axes = [g for g in 1:nd if !dropped[g]]
    # A 2D image keeps the spacing of the slice axis as its third entry.
    third = length(image_axes) == 2 && nd == 2 && !dropped[3] ? spacing[3] : nothing
    kept_spacing = [spacing[g] for g in image_axes]
    isnothing(third) || push!(kept_spacing, third)
    # Kept image axes first, so that the first columns of the orientation stay the image's own axes.
    if !isnothing(orientation) && length(image_axes) < nd
        orientation = orientation[:, vcat(image_axes, setdiff(1:3, image_axes))]
    end
    h.orientation = orientation
    # One image axis left is a profile, which the header's 2- or 3-axis geometry cannot describe.
    keep_geometry = length(image_axes) >= 2
    h.spacing = keep_geometry && all(!isnothing, kept_spacing) ? Tuple(kept_spacing) : nothing
    kept_fov = [fov[g] for g in image_axes]
    h.fov = keep_geometry && !isnothing(fov0) && all(!isnothing, kept_fov) ? Tuple(kept_fov) : nothing
    kept_size = [voxel_size[g] for g in image_axes]
    differs = kept_size != kept_spacing[1:length(kept_size)]
    h.voxel_size = keep_geometry && all(!isnothing, kept_size) && (!isnothing(vs0) || differs) ? Tuple(kept_size) : nothing
    h.offset = offset
    return h, length(image_axes)
end

# An index as an integer, a range or `:`, or `nothing` when it has no constant step.
_regular_index(idx::Union{Integer, Colon, AbstractRange{<:Integer}}) = idx
_regular_index(idx::AbstractVector{Bool}) = _regular_index(findall(idx))
function _regular_index(idx::AbstractVector{<:Integer})
    isempty(idx) && return nothing
    length(idx) == 1 && return first(idx):first(idx)
    s = idx[2] - idx[1]
    s != 0 && all(i -> idx[i + 1] - idx[i] == s, 1:(length(idx) - 1)) || return nothing
    return range(first(idx); step = s, length = length(idx))
end
_regular_index(idx) = nothing

# A missing orientation is the identity, as on export.
_orientation(orientation) = something(orientation, Matrix{Float64}(I, 3, 3))
_flipped(orientation, axis) = (o = copy(_orientation(orientation)); o[:, axis] .*= -1; o)

# `offset` moved along image axis `axis` to the first index of `idx`.
function _moved(offset, orientation, axis, idx, spacing)
    (isnothing(offset) || isnothing(spacing)) && return nothing
    return offset .+ _orientation(orientation)[:, axis] .* ((first(idx) - 1) * spacing)
end

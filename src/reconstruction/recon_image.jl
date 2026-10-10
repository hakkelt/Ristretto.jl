"""
    ReconImage(data, header = Header(); components = nothing, spatial_ndims) <: AbstractArray

The result of [`reconstruct`](@ref): the image `data` (a `NamedDimsArray` when the k-space was
named, on the host or a device) together with a copy of the acquisition's [`header`](@ref), which
carries the geometry (`fov`, `spacing`, `slice_spacing`, `orientation`, `offset`), the sequence
parameters and the tags (see [`Header`](@ref)). The first `spatial_ndims` axes of `data` are the
image axes. When it is not given, it is the length of the header's `spacing` or `fov`, or else
the number of leading axes named `:x`, `:y`, `:z`.

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
ReconImage(data::AbstractArray, header = Header(); components = nothing, spatial_ndims::Integer = _guess_spatial_ndims(data, _to_header(header))) =
    ReconImage(data, _to_header(header), components, spatial_ndims)

# `ReconImage`'s real properties; a component of one of these names would be unreachable through
# `img.<name>`, so `check_components` rejects it up front.
const _RECON_IMAGE_RESERVED_NAMES = (:data, :header, :components)

function _guess_spatial_ndims(data, h::Header)
    sp, fov = h.spacing, h.fov
    isnothing(sp) || return min(length(sp), ndims(data))
    isnothing(fov) || return min(length(fov), ndims(data))
    data isa NamedDimsArray || return min(ndims(data), 2)
    names = dimnames(data)
    n = 0
    while n < min(3, length(names)) && names[n + 1] === (:x, :y, :z)[n + 1]
        n += 1
    end
    return n
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

NamedDims.unname(img::ReconImage) = unname(parent(img))
NamedDims.dimnames(img::ReconImage) = dimnames(parent(img))
NamedDims.dimnames(img::ReconImage, d::Integer) = dimnames(parent(img), d)
NamedDims.NamedDimsArray(img::ReconImage) =
    parent(img) isa NamedDimsArray ? parent(img) : throw(ArgumentError("this ReconImage has no dimension names"))

Adapt.adapt_structure(to, img::ReconImage) =
    ReconImage(_adapt_any(to, parent(img)), header(img), _map_components(c -> Adapt.adapt(to, c), img), _spatial_ndims(img))
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

`img` without its components: the same image data and header, so that the components can be
garbage-collected once nothing else refers to them.
"""
drop_components(img::ReconImage) = ReconImage(parent(img), header(img), nothing, _spatial_ndims(img))

"""
    total_image(img::ReconImage)

The image itself, without the header: the sum of the components for a reconstruction with
[`Component`](@ref)s.
"""
total_image(img::ReconImage) = parent(img)

function Base.getproperty(img::ReconImage, name::Symbol)
    (name === :data || name === :header || name === :components) && return getfield(img, name)
    c = getfield(img, :components)
    !isnothing(c) && haskey(c, name) && return c[name]
    throw(ArgumentError("ReconImage has no property `$name`; available properties: $(join(propertynames(img), ", "))."))
end

function Base.propertynames(img::ReconImage, ::Bool = false)
    c = getfield(img, :components)
    return isnothing(c) ? (:data, :header, :components) : (:data, :header, :components, keys(c)...)
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
    h, nd = _slice_geometry(header(img), dimnames(data)[1:_spatial_ndims(img)], sel)
    return ReconImage(out, h, comps, nd)
end

# The header of a part of an image whose image axes are named `spatial`: for each image axis the
# offset moves to the first voxel kept and the axis shrinks to the voxels kept, or is dropped when
# an integer selects one; `:z` of a multi-slice 2D image moves the offset by the slice spacing. Any
# other axis leaves the geometry alone. Returns the header and the number of image axes kept.
function _slice_geometry(h0::Header, spatial, sel::NamedTuple)
    h = copy(h0)
    nd = length(spatial)
    fov0, spacing0 = h.fov, h.spacing
    fov = isnothing(fov0) || length(fov0) != nd ? nothing : collect(fov0)
    spacing = isnothing(spacing0) || length(spacing0) != nd ? nothing : collect(spacing0)
    offset = isnothing(h.offset) ? nothing : collect(h.offset)
    orientation = h.orientation
    keep = collect(1:nd)
    for (name, idx) in pairs(sel)
        axis = findfirst(==(name), spatial)
        if axis === nothing
            if name === :z && nd == 2
                offset = _moved(offset, orientation, 3, idx, h.slice_spacing)
            end
            continue
        end
        a = findfirst(==(axis), keep)
        offset = _moved(offset, orientation, axis, idx, isnothing(spacing) ? nothing : spacing[a])
        if idx isa Integer
            isnothing(spacing) || (h.slice_thickness = spacing[a])
            isnothing(fov) || deleteat!(fov, a)
            isnothing(spacing) || deleteat!(spacing, a)
            deleteat!(keep, a)
        elseif idx isa AbstractRange
            isnothing(spacing) || (spacing[a] *= step(idx))
            isnothing(fov) || isnothing(spacing) || (fov[a] = length(idx) * spacing[a])
        else
            offset = nothing
        end
    end
    # Kept axes first, so that the first columns of the orientation stay the image's own axes.
    if !isnothing(orientation) && length(keep) < nd
        h.orientation = orientation[:, vcat(keep, setdiff(1:3, keep))]
    end
    # One image axis left is a profile, which the header's 2- or 3-axis geometry cannot describe.
    h.fov = isnothing(fov) || length(fov) < 2 ? nothing : fov
    h.spacing = isnothing(spacing) || length(spacing) < 2 ? nothing : spacing
    h.offset = offset
    return h, length(keep)
end

# `offset` moved along image axis `axis` to the first index of `idx`.
_moved(::Nothing, orientation, axis, idx, spacing) = nothing
function _moved(offset, orientation, axis, idx, spacing)
    (isnothing(orientation) || isnothing(spacing)) && return nothing
    first_index = idx isa Integer ? idx : idx isa AbstractRange ? first(idx) : return nothing
    return offset .+ orientation[:, axis] .* ((first_index - 1) * spacing)
end

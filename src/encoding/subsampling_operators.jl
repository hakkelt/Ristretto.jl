"""
Subsampling operators for accelerated MRI reconstruction.

This module provides functions for creating operators that model undersampled
k-space acquisition patterns. These operators select subsets of k-space data
according to various sampling strategies used in compressed sensing MRI.
"""

"""
	get_subsampled_fourier_operator(info::CartesianAcquisitionInfo)
    get_subsampled_fourier_operator(subsampled_ksp, img_size, subsampling; shifted_kspace_dims=(), shifted_image_dims=(), threaded=true, fast_planning=false)

Create a combined Fourier transform and subsampling operator.

# Arguments when using explicit parameters
- `subsampled_ksp`: Subsampled k-space data array
- `img_size`: Size of the full image (2 or 3 element tuple)
- `subsampling`: Subsampling pattern used to generate `subsampled_ksp`
- `shifted_kspace_dims::Tuple=()`: Dimensions in k‑space where the DC is at the first index (useful for pre‑shifted data)
- `shifted_image_dims::Tuple=()`: Image dimensions requiring fft shift (equivalent to an kspace-domain sign-alternation)
- `threaded::Bool=true`: Whether to use multi-threading
- `fast_planning::Bool=false`: Whether to use fast FFTW planning

# Returns
- Composed operator `𝒫 * ℱ` where `ℱ` is Fourier transform and `𝒫` is subsampling

# Method Variants
- **Explicit parameters**: Requires `subsampled_ksp`, `img_size`, and `subsampling`
- **CartesianAcquisitionInfo**: Extracts necessary parameters from the acquisition struct

# Details
This function creates an operator that:
1. Takes an image as input
2. Applies the Fourier transform to get full k-space
3. Applies subsampling to get the observed k-space data

The adjoint operation reconstructs an image from subsampled k-space data.
"""
function get_subsampled_fourier_operator(
        subsampled_ksp, img_size, subsampling;
        shifted_kspace_dims::Tuple = (),
        shifted_image_dims::Tuple = (),
        threaded::Bool = true,
        fast_planning::Bool = false
    )
    ksp, 𝒫 = _build_subsampling_context(subsampled_ksp, img_size, subsampling; threaded)
    is3D = length(img_size) == 3
    ℱ = ksp isa NamedDimsArray ?
        get_fourier_operator(ksp; shifted_kspace_dims, shifted_image_dims, threaded, fast_planning) :
        get_fourier_operator(ksp, is3D; shifted_kspace_dims, shifted_image_dims, threaded, fast_planning)
    return 𝒫 * ℱ
end

function get_subsampled_fourier_operator(info::CartesianAcquisitionInfo; threaded::Bool = true, fast_planning::Bool = false)
    @argcheck !isnothing(info.kspace_data) "The provided CartesianAcquisitionInfo does not contain k-space data, which is required to build the subsampled Fourier operator."
    @argcheck !isnothing(info.subsampling) "CartesianAcquisitionInfo must include a subsampling pattern to build a subsampled Fourier operator."
    return get_subsampled_fourier_operator(info.kspace_data, info.image_size, info.subsampling; shifted_kspace_dims = info.shifted_kspace_dims, shifted_image_dims = info.shifted_image_dims, threaded, fast_planning)
end


"""
	get_subsampling_operator(subsampled_ksp, img_size, subsampling; threaded=true)
	get_subsampling_operator(info::CartesianAcquisitionInfo; threaded=true)

Create the subsampling operator 𝒫 that maps full k-space to a given
subsampled layout. This is useful when you need 𝒫 separately or want to
compose it with other operators manually. For a combined `𝒫 * ℱ` operator,
use `get_subsampled_fourier_operator`.

# Arguments
- `subsampled_ksp`: Subsampled k-space array (Array or NamedDimsArray).
  Used to determine batch dimensions and, for named arrays, validate
  dimension names.
- `img_size`: Full image size as `(nx, ny)` or `(nx, ny, nz)`.
- `subsampling`: Subsampling pattern that produced `subsampled_ksp`.
  Supports boolean masks and tuples mixing `Colon`, boolean masks, and ranges.
- `info::CartesianAcquisitionInfo`: Alternative API that takes configuration from a
	validated acquisition struct (must contain `image_size` and `subsampling`).
- `threaded::Bool=true`: Whether the batched form (one `GetIndex` per coil, slice or frame)
  applies its elements in parallel.

# Returns
- `𝒫`: A `GetIndex` or `BatchOp{GetIndex}` — or, when `subsampling` is an array of per-frame specs
  selecting *different numbers of samples*, a `VCAT` of per-frame `GetIndex`es whose codomain is an
  `ArrayPartition` (see [`PartitionedKSpace`](@ref)). For NamedDims inputs, a
  `NamedDimsOp` wrapping the un-named 𝒫 is returned to preserve
  dimension names; a partitioned codomain carries no names, since its blocks are separate arrays
  rather than axes of one.

# Details
- For NamedDims input, validates that `dimnames(subsampled_ksp)` matches the
  names implied by the subsampling pattern and the full k-space layout.
- Batch dimensions (beyond the spatial dims) are preserved; 𝒫 becomes a batch
  operator when needed.

# Examples
```julia
using Ristretto

# 2D mask, array input
ksp_full = rand(ComplexF32, 64, 64, 8)
mask = rand(Bool, 64, 64)
ksp_sub = ksp_full[mask, :]
𝒫 = get_subsampling_operator(ksp_sub, (64, 64), mask)

# 2D NamedDims input
using NamedDims
ksp_nd = NamedDimsArray{(:kxy, :coil)}(ksp_sub)
𝒫_nd = get_subsampling_operator(ksp_nd, (64, 64), mask)

# Via CartesianAcquisitionInfo
info = CartesianAcquisitionInfo(ksp_sub; is3D=false, image_size=(64, 64), subsampling=mask)
𝒫_info = get_subsampling_operator(info)
```
"""
function get_subsampling_operator(subsampled_ksp, img_size, subsampling; threaded::Bool = true)
    _, 𝒫 = _build_subsampling_context(subsampled_ksp, img_size, subsampling; threaded)
    return 𝒫
end

function get_subsampling_operator(acq_info::CartesianAcquisitionInfo; threaded::Bool = true)
    @argcheck !isnothing(acq_info.subsampling) "CartesianAcquisitionInfo must include a subsampling pattern"
    return get_subsampling_operator(acq_info.kspace_data, acq_info.image_size, acq_info.subsampling; threaded)
end

# -------- Type definitions for different subsampling patterns --------

const _1D_subsampling_type = Union{
    Colon, OrdinalRange{Int}, AbstractArray{Bool, 1}, AbstractVector{Int},
}

const _2D_subsampling_type = Union{
    Tuple{<:AbstractArray{Bool, 2}},
    Tuple{<:AbstractVector{Int}},
    Tuple{<:AbstractVector{CartesianIndex{2}}},
    Tuple{<:_1D_subsampling_type, <:_1D_subsampling_type},
}

const _3D_subsampling_type = Union{
    Tuple{<:AbstractArray{Bool, 3}},
    Tuple{<:AbstractVector{Int}},
    Tuple{<:AbstractVector{CartesianIndex{3}}},
    Tuple{<:_1D_subsampling_type, <:AbstractArray{Bool, 2}},
    Tuple{<:_1D_subsampling_type, <:AbstractVector{Int}},
    Tuple{<:_1D_subsampling_type, <:AbstractVector{CartesianIndex{2}}},
    Tuple{<:_1D_subsampling_type, <:_1D_subsampling_type, <:_1D_subsampling_type},
}

# -------- Internal helper functions --------

function _check_ksp_dimnames(ksp_dimnames, subs::Nothing, is3D, img_size)
    @argcheck :kx ∈ ksp_dimnames && :ky ∈ ksp_dimnames "k-space must have :kx and :ky dimensions"
    @argcheck ksp_dimnames[1] == :kx && ksp_dimnames[2] == :ky "first dims must be :kx, :ky"
    return if is3D
        @argcheck :kz ∈ ksp_dimnames && ksp_dimnames[3] == :kz "third dim must be :kz for 3D"
    end
end

function _check_ksp_dimnames(ksp_dimnames, subs::_2D_subsampling_type, is3D, img_size::Tuple{Int, Int})
    expected_prefix = if subs isa Tuple{<:_1D_subsampling_type, <:_1D_subsampling_type}
        (:kx, :ky)
    else
        (:kxy,)
    end
    if :coil ∈ ksp_dimnames
        expected_prefix = (expected_prefix..., :coil)
    end
    if :z ∈ ksp_dimnames
        expected_prefix = (expected_prefix..., :z)
    end
    @argcheck ksp_dimnames[1:length(expected_prefix)] == expected_prefix "k-space dimension names must start with $(expected_prefix) for 2D subsampling"
    if subs isa Tuple{<:_1D_subsampling_type, <:_1D_subsampling_type}
        @argcheck :kx ∈ ksp_dimnames "k-space must have :kx dimension"
        @argcheck :ky ∈ ksp_dimnames "k-space must have :ky dimension"
    else
        @argcheck :kxy ∈ ksp_dimnames "k-space must have :kxy dimension"
    end
    @argcheck length(img_size) == 2 "image_size must be length 2 for 2D subsampling"
    @argcheck !is3D "is3D must be false for 2D subsampling"
    return nothing
end

function _check_ksp_dimnames(ksp_dimnames, subs::_3D_subsampling_type, is3D, img_size::Tuple{Int, Int, Int})
    expected_prefix = if length(subs) == 3
        (:kx, :ky, :kz)
    elseif length(subs) == 2 && subs[1] isa _1D_subsampling_type
        (:kx, :kyz)
    elseif length(subs) == 2 && subs[2] isa _1D_subsampling_type
        (:kxy, :kz)
    else
        (:kxyz,)
    end
    if :coil ∈ ksp_dimnames
        expected_prefix = (expected_prefix..., :coil)
    end
    @argcheck ksp_dimnames[1:length(expected_prefix)] == expected_prefix "k-space dimension names must start with $(expected_prefix) for 3D subsampling"
    if subs isa Tuple{<:_1D_subsampling_type, <:_1D_subsampling_type, <:_1D_subsampling_type}
        @argcheck :kx ∈ ksp_dimnames "k-space must have :kx dimension"
        @argcheck :ky ∈ ksp_dimnames "k-space must have :ky dimension"
        @argcheck :kz ∈ ksp_dimnames "k-space must have :kz dimension"
    elseif length(subs) == 2 && subs[2] isa _1D_subsampling_type
        @argcheck :kxy ∈ ksp_dimnames "k-space must have :kxy dimension"
        @argcheck :kz ∈ ksp_dimnames "k-space must have :kz dimension"
    elseif length(subs) == 2 && subs[1] isa _1D_subsampling_type
        @argcheck :kx ∈ ksp_dimnames "k-space must have :kx dimension"
        @argcheck :kyz ∈ ksp_dimnames "k-space must have :kyz dimension"
    else
        @argcheck :kxyz ∈ ksp_dimnames "k-space must have :kxyz dimension"
    end
    @argcheck length(img_size) == 3 "image_size must be length 3 for 3D subsampling"
    @argcheck is3D "is3D must be true for 3D subsampling"
    return @argcheck !(:z ∈ ksp_dimnames) "3D subsampling cannot have :z dimension"
end

function _check_ksp_dimnames(ksp_dimnames, subs::AbstractArray, is3D, img_size)
    @argcheck !isempty(subs) "subsampling array must not be empty"
    ref_subs = _normalize_subsampling(first(subs))
    return _check_ksp_dimnames(ksp_dimnames, ref_subs, is3D, img_size)
end

function _get_dimnames_from_subsampling(ksp_dimnames, ::Tuple{Int, Int}, subsampling::_2D_subsampling_type)
    if length(subsampling) == 2
        return (:kx, :ky, ksp_dimnames[3:end]...)
    else
        return (:kxy, ksp_dimnames[3:end]...)
    end
end

function _get_dimnames_from_subsampling(ksp_dimnames, ::Tuple{Int, Int, Int}, subsampling::_3D_subsampling_type)
    if length(subsampling) == 3
        return (:kx, :ky, :kz, ksp_dimnames[4:end]...)
    elseif length(subsampling) == 2 && subsampling[1] isa _1D_subsampling_type
        return (:kx, :kyz, ksp_dimnames[4:end]...)
    elseif length(subsampling) == 2 && subsampling[2] isa _1D_subsampling_type
        return (:kxy, :kz, ksp_dimnames[4:end]...)
    else
        return (:kxyz, ksp_dimnames[4:end]...)
    end
end

function _get_dimnames_from_subsampling(ksp_dimnames, img_size, subsampling::AbstractArray)
    @argcheck !isempty(subsampling) "subsampling array must not be empty"
    return _get_dimnames_from_subsampling(
        ksp_dimnames,
        img_size,
        _normalize_subsampling(first(subsampling)),
    )
end

"""
    _device_batched_getindex(ksp, img_size, subsampling)

The subsampling of a device k-space with batch axes (coils, slices, frames) as one `GetIndex` over
the whole array, the batch axes taken whole, instead of a `BatchOp` of one `GetIndex` per batch
element. The two select the same samples in the same order; on a device the batch runs its
elements one after another, each a kernel launch of its own, which on a 128×128 image with 8 coils
and 30 frames costs more than the gather itself. A linear index into the Fourier grid becomes the
`CartesianIndex` it stands for, so that the trailing `:`s index the batch axes rather than extend
the linear index.
"""
function _device_batched_getindex(ksp, img_size, subsampling::Tuple)
    nbatch = ndims(ksp) - length(img_size)
    return GetIndex(ksp, (_fourier_index(img_size, subsampling)..., ntuple(_ -> Colon(), nbatch)...))
end

_fourier_index(img_size, subsampling::Tuple) = _plane_index(img_size, subsampling)
_fourier_index(img_size, subsampling::Tuple{AbstractVector{Int}}) = (CartesianIndices(img_size)[only(subsampling)],)

# A 3D spec `(kx, v)` with `v` a linear index into the ky–kz plane, as the `CartesianIndex`es it
# stands for. `GetIndex` checks its indices axis by axis, so an integer index running over two
# axes is out of bounds for ky, and any trailing batch `:` would extend it besides.
_plane_index(img_size, subsampling::Tuple) = subsampling
function _plane_index(img_size::NTuple{3, Int}, subsampling::Tuple{_1D_subsampling_type, AbstractVector{Int}})
    return (subsampling[1], CartesianIndices(img_size[2:3])[subsampling[2]])
end

"""
    _device_spreading_getindex(ksp, img_size, subsampling::AbstractArray, prefix_batch_dims)

The subsampling of a device k-space by one spec per element of its trailing axes (one mask per
frame, say) as a single `GetIndex` of linear indices into the whole array, reshaped to
`(samples of one element..., size(subsampling)...)`: the layout the `BatchOp` of one `GetIndex`
per element gives, with the same samples in the same order. The indices are worked out on the
host, one `Int` per sample, and moved to the device once; the `BatchOp` would launch a gather
per element on every apply, and a fill and a scatter per element on every adjoint.
"""
function _device_spreading_getindex(ksp, img_size, subsampling::AbstractArray, prefix_batch_dims::Int)
    linear = LinearIndices(size(ksp))
    prefix = ntuple(_ -> Colon(), prefix_batch_dims)
    indices = stack(CartesianIndices(subsampling)) do I
        spec = _normalize_subsampling(subsampling[I])
        linear[_fourier_index(img_size, spec)..., prefix..., Tuple(I)...]
    end
    return reshape(GetIndex(ksp, (vec(indices),)), size(indices)...)
end

function _get_subsampling_operator(ksp, img_size::Tuple{Int, Int}, subsampling::_2D_subsampling_type; threaded::Bool)
    @argcheck length(img_size) == 2 "img_size must be a 2-element tuple for 2D subsampling"
    @argcheck img_size == size(ksp)[1:2] DimensionMismatch
    if ndims(ksp) > length(img_size)
        batch_dims = size(ksp)[3:end]
        _is_device(ksp) && return _device_batched_getindex(ksp, img_size, subsampling)
        ksp_view = @view ksp[:, :, fill(1, length(batch_dims))...]
        𝒫 = GetIndex(ksp_view, subsampling)
        return BatchOp(𝒫, batch_dims; threaded)
    else
        return GetIndex(ksp, subsampling)
    end
end

function _get_subsampling_operator(ksp, img_size::Tuple{Int, Int, Int}, subsampling::_3D_subsampling_type; threaded::Bool)
    @argcheck length(img_size) == 3 "img_size must be a 3-element tuple for 3D subsampling"
    @argcheck img_size == size(ksp)[1:3] DimensionMismatch
    subsampling = _plane_index(img_size, subsampling)
    if ndims(ksp) > length(img_size)
        batch_dims = size(ksp)[4:end]
        _is_device(ksp) && return _device_batched_getindex(ksp, img_size, subsampling)
        ksp_view = @view ksp[:, :, :, fill(1, length(batch_dims))...]
        𝒫 = GetIndex(ksp_view, subsampling)
        return BatchOp(𝒫, batch_dims; threaded)
    else
        return GetIndex(ksp, subsampling)
    end
end

function _get_subsampling_operator(ksp, img_size, subsampling::AbstractArray; threaded::Bool)
    @argcheck !isempty(subsampling) "subsampling array must not be empty"
    fourier_dims = length(img_size)
    spreading_dims = ndims(subsampling)
    @argcheck size(ksp)[1:fourier_dims] == img_size DimensionMismatch
    nonspatial_dims = size(ksp)[(fourier_dims + 1):end]
    @argcheck length(nonspatial_dims) >= spreading_dims "k-space must have enough dimensions to match the subsampling array"

    spreading_start = nothing
    for start in 1:(length(nonspatial_dims) - spreading_dims + 1)
        stop = start + spreading_dims - 1
        if nonspatial_dims[start:stop] == size(subsampling)
            spreading_start = start
            break
        end
    end
    @argcheck !isnothing(spreading_start) "could not align subsampling array dimensions with the nonspatial k-space dimensions"

    prefix_batch_dims = spreading_start - 1
    suffix_batch_dims = length(nonspatial_dims) - (spreading_start + spreading_dims - 1)

    # A per-element spec array may select a *different number of samples* per element. A dense
    # k-space array cannot hold that (the sample axis has one length for the whole array), so the
    # measurement side becomes an `ArrayPartition` and the operator a `VCAT` of per-element
    # `GetIndex`es rather than a `BatchOp`. Detected here, from the specs themselves, so the
    # equal-count case keeps exactly the dense path it had.
    if _has_unequal_sample_counts(ksp, img_size, subsampling)
        return _get_partitioned_subsampling_operator(ksp, img_size, subsampling)
    end

    if _is_device(ksp) && suffix_batch_dims == 0
        return _device_spreading_getindex(ksp, img_size, subsampling, prefix_batch_dims)
    end

    op_indices = CartesianIndices(subsampling)
    first_index = first(op_indices)

    first_view = @view ksp[
        fill(:, fourier_dims)...,
        fill(:, prefix_batch_dims)...,
        Tuple(first_index)...,
        fill(:, suffix_batch_dims)...,
    ]
    first_op = _get_subsampling_operator(
        first_view,
        img_size,
        _normalize_subsampling(subsampling[first_index]);
        threaded,
    )
    operators = Array{typeof(first_op)}(undef, size(subsampling))
    operators[first_index] = first_op

    for index in Iterators.drop(op_indices, 1)
        local_view = @view ksp[
            fill(:, fourier_dims)...,
            fill(:, prefix_batch_dims)...,
            Tuple(index)...,
            fill(:, suffix_batch_dims)...,
        ]
        operators[index] = _get_subsampling_operator(
            local_view,
            img_size,
            _normalize_subsampling(subsampling[index]);
            threaded,
        )
    end

    spreading_dim_count = ndims(subsampling)
    n_in = ndims(first_op, 2)::Int + spreading_dim_count
    n_out = ndims(first_op, 1)::Int + spreading_dim_count
    domain_mask = ntuple(
        i -> i <= ndims(first_op, 2)::Int ? :_ : :s,
        n_in,
    )
    codomain_mask = ntuple(
        i -> i <= ndims(first_op, 1)::Int ? :_ : :s,
        n_out,
    )
    return BatchOp(operators, domain_mask => codomain_mask; threaded)
end

# How many samples one spec selects out of the full Fourier grid. Only the Fourier dimensions
# matter: the coil and batch axes are shared by every element of the spec array.
function _spec_sample_count(img_size, spec)
    idx = _normalize_subsampling(spec)
    return prod(AbstractOperators.get_dim_out(img_size, idx...))
end

function _has_unequal_sample_counts(ksp, img_size, subsampling::AbstractArray)
    isempty(subsampling) && return false
    counts = map(spec -> _spec_sample_count(img_size, spec), subsampling)
    return !all(==(first(counts)), counts)
end

_has_unequal_sample_counts(ksp, img_size, subsampling) = false
# A single spec that happens to be an array (a mask, an index vector) is one pattern for the whole
# acquisition, not an array of per-frame patterns — the same distinction `_get_subsampling_operator`
# makes by dispatching the 1D/2D/3D spec types ahead of the spec-array method.
_has_unequal_sample_counts(ksp, img_size, ::_1D_subsampling_type) = false

"""
    _ragged_subsampling_dim(img_size, subsampling)

Which k-space dimension the per-frame specs disagree on. Exactly one is allowed to vary: the frames
share a k-space layout apart from how many samples they take along one axis, which is what makes the
data a partition of same-shaped blocks rather than an unrelated collection.
"""
function _ragged_subsampling_dim(img_size, subsampling::AbstractArray)
    shapes = map(spec -> AbstractOperators.get_dim_out(img_size, _normalize_subsampling(spec)...), subsampling)
    reference = first(shapes)
    differing = findall(d -> any(shape -> shape[d] != reference[d], shapes), 1:length(reference))
    @argcheck length(differing) == 1 "per-frame subsampling specs may differ in exactly one k-space dimension; these differ in $(isempty(differing) ? "none" : differing)"
    return only(differing)
end

# The unequal-count operator: one `GetIndex` per frame, reaching into the *full* dense k-space and
# picking both that frame's slab and its own mask, stacked by a `VCAT`. Its codomain is an
# `ArrayPartition` — one array per frame, each with its own sample count — which is exactly the
# storage `PartitionedKSpace` holds. No new operator type is needed, and `BatchOp`'s equal-size
# assertions stay in place, because this case never reaches `BatchOp`.
function _get_partitioned_subsampling_operator(ksp, img_size, subsampling)
    @argcheck subsampling isa AbstractVector "unequal per-frame sample counts are supported only for a Vector of subsampling specs, one per frame (got a $(ndims(subsampling))-dimensional spec array)"
    fourier_dims = length(img_size)
    nonspatial = size(ksp)[(fourier_dims + 1):end]
    @argcheck !isempty(nonspatial) && nonspatial[end] == length(subsampling) "a Vector of subsampling specs with unequal sample counts must span the last k-space dimension (k-space non-Fourier dimensions $(nonspatial), $(length(subsampling)) specs)"
    prefix_batch_dims = length(nonspatial) - 1
    operators = map(enumerate(subsampling)) do (frame, spec)
        idx = (
            _normalize_subsampling(spec)...,
            ntuple(_ -> Colon(), prefix_batch_dims)...,
            frame,
        )
        GetIndex(ksp, idx)
    end
    return VCAT(operators...)
end

_get_subsampled_dims_count(::AbstractArray{Bool}) = 1
_get_subsampled_dims_count(::AbstractVector{Int}) = 1
_get_subsampled_dims_count(::AbstractVector{<:CartesianIndex}) = 1
_get_subsampled_dims_count(::Colon) = 1
_get_subsampled_dims_count(::OrdinalRange) = 1
_get_subsampled_dims_count(subsampling::Tuple) = sum(_get_subsampled_dims_count.(subsampling))
function _get_subsampled_dims_count(subsampling::AbstractArray)
    dim_counts = _get_subsampled_dims_count.(subsampling)
    if all(==(dim_counts[1]), dim_counts)
        return dim_counts[1]
    else
        return error("Inconsistent subsampling dimensions inferred: $(dim_counts)")
    end
end

_get_img_size_from_subsampling(subsampling::AbstractArray{Bool}, ksp) = size(subsampling)
_get_img_size_from_subsampling(::AbstractVector{<:Integer}, ksp) = nothing
_get_img_size_from_subsampling(::AbstractVector{CartesianIndex{N}}, ksp) where {N} = nothing

function _get_img_size_from_subsampling(subsampling::Tuple, ksp)
    img_size = ()
    dim_counter = 1
    for subs in subsampling
        if subs isa Colon
            img_size = (img_size..., size(ksp, dim_counter)...)
            dim_counter += 1
        elseif subs isa AbstractArray{Bool}
            img_size = (img_size..., size(subs)...)
            dim_counter += ndims(subs)
        else
            return nothing
        end
    end
    return img_size
end

function _get_img_size_from_subsampling(subsampling::AbstractArray, ksp)
    guessed_sizes = tuple([_get_img_size_from_subsampling(subs, ksp) for subs in subsampling]...)
    if any(isnothing, guessed_sizes)
        return nothing
    elseif all(==(guessed_sizes[1]), guessed_sizes)
        return guessed_sizes[1]
    else
        return error("Inconsistent image sizes inferred from subsampling masks: $(guessed_sizes)")
    end
end

# Allocate an (uninitialized) full-size k-space array matching the layout implied by the
# subsampled data — used as a planning template so no adjoint apply is needed.
function _full_kspace_template(subsampled_ksp, img_size, subsampling)
    @argcheck 2 ≤ length(img_size) ≤ 3 "img_size must be either length 2 or 3"
    batch_dims_start = _get_subsampled_dims_count(subsampling) + 1
    ksp_size = (img_size..., size(subsampled_ksp)[batch_dims_start:end]...)
    ksp = similar(subsampled_ksp, ksp_size)
    if subsampled_ksp isa NamedDimsArray
        batch_dim_names = dimnames(subsampled_ksp)[batch_dims_start:end]
        full_dimnames = length(img_size) == 3 ?
            (:kx, :ky, :kz, batch_dim_names...) :
            (:kx, :ky, batch_dim_names...)
        expected_subs_dimnames = _get_dimnames_from_subsampling(full_dimnames, img_size, subsampling)
        @argcheck dimnames(subsampled_ksp) == expected_subs_dimnames
        ksp = NamedDimsArray{full_dimnames}(unname(ksp))
    end
    return ksp
end

# The *full* k-space is dense even when the measured one is not — every frame has the same full
# grid; only the selected samples differ. So the template stays an ordinary array, and the ragged
# dimension is never asked for.
function _full_kspace_template(subsampled_ksp::PartitionedKSpace, img_size, subsampling)
    @argcheck 2 ≤ length(img_size) ≤ 3 "img_size must be either length 2 or 3"
    batch_dims_start = _get_subsampled_dims_count(subsampling) + 1
    batch_sizes = _ksp_trailing_size(subsampled_ksp, batch_dims_start)
    ksp = similar(unname(first(parts(subsampled_ksp))), (img_size..., batch_sizes...))
    if !isnothing(subsampled_ksp.dimnames)
        batch_dim_names = dimnames(subsampled_ksp)[batch_dims_start:end]
        full_dimnames = length(img_size) == 3 ?
            (:kx, :ky, :kz, batch_dim_names...) :
            (:kx, :ky, batch_dim_names...)
        expected_subs_dimnames = _get_dimnames_from_subsampling(full_dimnames, img_size, subsampling)
        @argcheck dimnames(subsampled_ksp) == expected_subs_dimnames
        ksp = NamedDimsArray{full_dimnames}(ksp)
    end
    return ksp
end

function _build_subsampling_context(subsampled_ksp, img_size, subsampling; threaded::Bool)
    ksp = _full_kspace_template(subsampled_ksp, img_size, subsampling)
    if ksp isa NamedDimsArray
        𝒫_unwrapped = _get_subsampling_operator(unname(ksp), img_size, subsampling; threaded)
        D = dimnames(ksp)
        # A partitioned codomain has no dimension names to carry: its blocks are separate arrays
        # of different sizes, not axes of one array. `nothing` says so, rather than a name tuple
        # that would describe a shape the codomain does not have.
        new_dimnames = _has_unequal_sample_counts(ksp, img_size, subsampling) ?
            nothing :
            _get_dimnames_from_subsampling(D, img_size, subsampling)
        𝒫 = NamedDimsOp{D, new_dimnames}(𝒫_unwrapped)
    else
        𝒫 = _get_subsampling_operator(ksp, img_size, subsampling; threaded)
    end
    return ksp, 𝒫
end

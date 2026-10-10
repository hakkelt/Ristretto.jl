"""
    CartesianAcquisitionInfo(
            kspace_data;
            is3D::Union{Bool,Nothing}=nothing,
            sensitivity_maps=nothing,
            image_size=nothing,
            subsampling=nothing,
            shifted_kspace_dims::Tuple=(),
            shifted_image_dims::Tuple=(),
            header=nothing,
    )

Configuration container for Cartesian MRI acquisition and encoding settings.

`header` is the metadata: a [`Header`](@ref), stored as given, or keywords for one as a
`NamedTuple` or another dictionary. An empty header is created when none is given.

`subsampling` may be a single pattern shared by the whole acquisition, or an *array of patterns*,
one per batch element (a different ky mask per frame). In the latter case `kspace_data` is a dense
array as long as every frame selects the same number of samples; when the counts differ it is a
[`PartitionedKSpace`](@ref) instead, holding one array per frame.
"""
struct CartesianAcquisitionInfo{K, I, S, Sub, SD, ID} <: AcquisitionInfo
    kspace_data::K
    is3D::Bool
    image_size::I
    sensitivity_maps::S
    subsampling::Sub
    shifted_kspace_dims::SD
    shifted_image_dims::ID
    header::Header

    function CartesianAcquisitionInfo(ksp, is3D, img_size, smaps, subs, sK, sI, hdr = nothing)
        if !isnothing(subs)
            if !isnothing(ksp)
                guessed = Ristretto._get_img_size_from_subsampling(subs, ksp)
                if isnothing(img_size)
                    @argcheck !isnothing(guessed) "image_size must be provided or be inferable from subsampling"
                    img_size = guessed
                elseif !isnothing(guessed)
                    @argcheck img_size == guessed "image_size must match subsampling implied size"
                end
            end
            subs = _normalize_subsampling(subs)
        end

        if isnothing(is3D)
            if !isnothing(img_size)
                @argcheck length(img_size) == 2 || length(img_size) == 3 "image_size must be length 2 or 3"
                is3D = length(img_size) == 3
            else
                is3D = _infer_is3D_from_ksp(ksp)
                @argcheck !isnothing(is3D) "is3D must be provided when non-NamedDimsArray k-space is used and image_size is not given"
            end
        end

        if isnothing(img_size)
            if !isnothing(ksp)
                img_size = size(ksp)[1:(is3D ? 3 : 2)]
            elseif !isnothing(smaps)
                img_size = size(smaps)[1:(is3D ? 3 : 2)]
            end
        end
        @argcheck !isnothing(img_size) "image_size must be provided or inferable from subsampling or sensitivity maps"

        if _has_dimnames(ksp)
            _check_ksp_dimnames(dimnames(ksp), subs, is3D, img_size)
        end

        if !isnothing(smaps)
            _check_smaps(smaps, ksp, subs, is3D, img_size)
            _check_same_storage(ksp, smaps, "sensitivity maps")
        end

        if sK != ()
            if sK isa Integer || sK isa Symbol
                sK = (sK,)
            end
            for d in sK
                @argcheck d isa Integer || d isa Symbol "shifted_kspace_dims must be Integer, Symbol, or Tuple of those"
                if d isa Integer
                    @argcheck d ∈ 1:(isnothing(ksp) ? length(img_size) : ndims(ksp)) "shifted_kspace_dims out of range"
                else
                    @argcheck _has_dimnames(ksp) "shifted_kspace_dims as Symbol requires NamedDimsArray k-space"
                    @argcheck d ∈ dimnames(ksp) "shifted_kspace_dims Symbol not found in k-space dimnames"
                end
            end
        end
        if sI != ()
            if sI isa Integer || sI isa Symbol
                sI = (sI,)
            end
            for d in sI
                @argcheck d isa Integer || d isa Symbol "shifted_image_dims must be Integer, Symbol, or Tuple of those"
                if d isa Integer
                    @argcheck d ∈ 1:length(img_size) "shifted_image_dims out of range"
                else
                    @argcheck _has_dimnames(ksp) "shifted_image_dims as Symbol requires NamedDimsArray k-space"
                    img_dimnames = (:x, :y, (is3D ? :z : ()), dimnames(ksp)[(is3D ? 4 : 3):end]...) |> filter(!=(()))
                    @argcheck d ∈ img_dimnames "shifted_image_dims Symbol not found in image dimnames"
                end
            end
        end

        hdr = _to_header(hdr)
        _check_geometry(hdr, img_size)
        return new{typeof(ksp), typeof(img_size), typeof(smaps), typeof(subs), typeof(sK), typeof(sI)}(
            ksp, is3D, img_size, smaps, subs, sK, sI, hdr
        )
    end
end

_normalize_subsampling(subs::Tuple) = subs
_normalize_subsampling(subs::Colon) = (subs,)
_normalize_subsampling(subs::OrdinalRange{Int}) = (subs,)
_normalize_subsampling(subs::AbstractArray{Bool}) = (subs,)
_normalize_subsampling(subs::AbstractVector{<:Integer}) = (subs,)
_normalize_subsampling(subs::AbstractVector{<:CartesianIndex}) = (subs,)
_normalize_subsampling(subs) = subs

CartesianAcquisitionInfo(
    kspace_data;
    is3D::Union{Bool, Nothing} = nothing,
    image_size = nothing,
    sensitivity_maps = nothing,
    subsampling = nothing,
    shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
    shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
    header = nothing,
) = CartesianAcquisitionInfo(kspace_data, is3D, image_size, sensitivity_maps, subsampling, shifted_kspace_dims, shifted_image_dims, header)

CartesianAcquisitionInfo(;
    kspace_data = nothing,
    is3D::Union{Bool, Nothing} = nothing,
    image_size = nothing,
    sensitivity_maps = nothing,
    subsampling = nothing,
    shifted_kspace_dims::Union{Tuple, Integer, Symbol} = (),
    shifted_image_dims::Union{Tuple, Integer, Symbol} = (),
    header = nothing,
) = CartesianAcquisitionInfo(kspace_data, is3D, image_size, sensitivity_maps, subsampling, shifted_kspace_dims, shifted_image_dims, header)


function _check_smaps(smaps, ksp, subs, is3D, img_size)
    # `length(subs)` is only the number of subsampled k-space axes when `subs` is the per-axis
    # tuple; a per-batch-element subsampling (a Vector of specs, one ky mask per frame) has
    # `length == nframes`. `_get_subsampled_dims_count` handles both.
    ksp_dims_count = isnothing(subs) ? (is3D ? 3 : 2) : _get_subsampled_dims_count(subs)
    if _has_dimnames(ksp)
        @argcheck smaps isa NamedDimsArray "sensitivity maps must be NamedDimsArray when k-space is NamedDimsArray"
        @argcheck :coil ∈ dimnames(ksp) ":coil dimension required in k-space when sensitivity maps are provided"
        if is3D
            @argcheck :z ∉ dimnames(ksp) "3D k-space must not have :z dimension"
            # kz is its own axis unless the subsampling joined it with ky (`:kyz`) or with kx and ky
            # (`:kxyz`); see `_get_subsampled_dimnames`.
            @argcheck any(∈((:kz, :kyz, :kxyz)), dimnames(ksp)) "3D k-space must have a :kz, :kyz or :kxyz dimension"
            @argcheck dimnames(smaps) == (:x, :y, :z, :coil) "sensitivity maps dimnames must be (:x, :y, :z, :coil) for 3D acquisition"
        elseif ndims(smaps) == 4
            @argcheck dimnames(smaps) == (:x, :y, :coil, :z) "sensitivity maps dimnames must be (:x, :y, :coil, :z) for 2D acquisition"
            @argcheck dimnames(ksp, ksp_dims_count + 1) == :coil "k-space coil dimension must be right after k-space dimensions"
            @argcheck dimnames(ksp, ksp_dims_count + 2) == :z "k-space slice dimension must be right after coil dimension for 2D multislice acquisition"
            @argcheck :kz ∉ dimnames(ksp) "2D k-space must not have :kz dimension"
        else
            @argcheck dimnames(smaps) == (:x, :y, :coil) "sensitivity maps dimnames must be (:x, :y, :coil) for 2D acquisition"
            @argcheck dimnames(ksp, ksp_dims_count + 1) == :coil "k-space coil dimension must be right after k-space dimensions"
            @argcheck :kz ∉ dimnames(ksp) "2D k-space must not have :kz dimension"
        end
    end
    return if !isnothing(ksp)
        @argcheck eltype(ksp) == eltype(smaps) "k-space and sensitivity maps eltype mismatch"
        if is3D
            @argcheck ndims(smaps) == 4 "sensitivity maps must be 4D for 3D acquisition"
            @argcheck size(smaps)[1:3] == img_size "sensitivity maps and image spatial dimensions size mismatch for 3D acquisition"
        elseif ndims(smaps) == 4
            @argcheck ndims(ksp) >= ksp_dims_count + 2 "k-space must have slice dimension when sensitivity maps are 4D for 2D acquisition"
            @argcheck size(smaps)[1:2] == img_size "sensitivity maps and image spatial dimensions size mismatch for 2D acquisition"
            @argcheck size(ksp, ksp_dims_count + 1) == size(smaps, 3) "k-space and sensitivity maps coil dimension size mismatch"
            @argcheck size(ksp, ksp_dims_count + 2) == size(smaps, 4) "k-space and sensitivity maps slice dimension size mismatch"
        else
            @argcheck size(smaps)[1:2] == img_size "sensitivity maps and image spatial dimensions size mismatch for 2D acquisition"
            @argcheck ndims(smaps) == 3 "sensitivity maps must be 3D for 2D acquisition"
            @argcheck ndims(ksp) >= ksp_dims_count + 1 "k-space must have coil dimension when sensitivity maps are 3D for 2D acquisition"
            @argcheck size(ksp, ksp_dims_count + 1) == size(smaps, 3) "k-space and sensitivity maps coil dimension size mismatch"
        end
    end
end

function _subsample_item_to_str(item)
    if item isa Tuple
        return "($(join(map(_subsample_item_to_str, item), ", ")))"
    elseif item isa Integer || item isa AbstractRange
        return string(item)
    elseif item isa AbstractVector
        return "Vector{$(eltype(item))}<$(join(size(item), "×"))>"
    elseif item isa AbstractArray
        return "Array{$(eltype(item))}<$(join(size(item), "×"))>"
    elseif item isa Colon
        return ":"
    else
        return "?"
    end
end

# The per-axis tuple form, `(:, mask)`.
function _subsampling_to_str(subs::Tuple)
    strs = map(_subsample_item_to_str, subs)
    return length(strs) == 1 ? strs[1] : "($(join(strs, ", ")))"
end
_subsampling_to_str(subs::AbstractArray{Bool}) = _subsample_item_to_str(subs)
# One spec per batch element (e.g. a different ky mask per frame): summarize as `19×(:, ...)`
# instead of printing all nineteen.
_subsampling_to_str(subs::AbstractArray) =
    "$(join(size(subs), "×"))×$(_subsampling_to_str(_normalize_subsampling(first(subs))))"

function _get_acq_info_meta(info::CartesianAcquisitionInfo)
    meta = String[]
    if info.kspace_data isa PartitionedKSpace
        # Its own `show` is the honest summary: the ragged axis printed as its per-frame counts
        # rather than as a single number it does not have.
        push!(meta, "kspace_data=$(info.kspace_data)")
    elseif !isnothing(info.kspace_data)
        if info.kspace_data isa NamedDimsArray
            dims = ["$dim: $s" for (dim, s) in zip(dimnames(info.kspace_data), size(info.kspace_data))]
            size_str = join(dims, ", ")
        else
            size_str = join(size(info.kspace_data), "×")
        end
        push!(meta, "kspace_data=Array{$(eltype(info.kspace_data))}<$size_str>")
    end
    push!(meta, "encoding=" * (info.is3D ? "3D" : "2D"))
    push!(meta, "image_size=$(join(info.image_size, "×"))")
    if !isnothing(info.sensitivity_maps)
        push!(meta, "sensitivity_maps=$(eltype(info.sensitivity_maps))<$(join(size(info.sensitivity_maps), "×"))>")
    end
    if !isnothing(info.subsampling)
        push!(meta, "subsampling=$(_subsampling_to_str(info.subsampling))")
    end
    if !isempty(info.shifted_kspace_dims)
        if length(info.shifted_kspace_dims) == 1
            shifted_kspace_dims = string(info.shifted_kspace_dims[1])
        else
            shifted_kspace_dims = "($(join(info.shifted_kspace_dims, ",")))"
        end
        push!(meta, "shifted_kspace_dims=$shifted_kspace_dims")
    end
    if !isempty(info.shifted_image_dims)
        if length(info.shifted_image_dims) == 1
            shifted_image_dims = string(info.shifted_image_dims[1])
        else
            shifted_image_dims = "($(join(info.shifted_image_dims, ",")))"
        end
        push!(meta, "shifted_image_dims=$shifted_image_dims")
    end
    return meta
end

function Base.show(io::IO, info::CartesianAcquisitionInfo)
    meta = _get_acq_info_meta(info)
    return print(io, "CartesianAcquisitionInfo(", join(meta, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", info::CartesianAcquisitionInfo)
    meta = _get_acq_info_meta(info)
    println(io, "CartesianAcquisitionInfo:")
    for (i, m) in enumerate(meta)
        m = replace(m, "=" => " = ", "_" => " ")
        print(io, "  - $m")
        if i < length(meta)
            println(io)
        end
    end
    return nothing
end

function _infer_is3D_from_ksp(ksp)
    if ksp isa NamedDimsArray
        return :kz ∈ dimnames(ksp)
    end
    return nothing
end

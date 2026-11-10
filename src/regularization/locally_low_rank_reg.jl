"""
	BlockNuclearNorm(λ, block_size, spatial_size, num_frames, num_batch, threaded; shift=:none, rng=Random.default_rng())

Proximable function implementing the sum of nuclear norms of the Casorati matrices of a set of
non-overlapping spatial blocks: `f(x) = λ ∑_b ‖𝓧_b‖_*`, where `𝓧_b` is the `(∏ block_size) × num_frames`
matrix formed by the voxels of block `b`.

This is the function behind [`LocallyLowRank`](@ref); it is applied to the image itself (identity operator),
because the block decomposition is a permutation of the voxels and the prox therefore decouples exactly into
one singular value thresholding per block.

`shift` controls where the tiling grid starts (see [`LocallyLowRank`](@ref)): `:none` anchors it at the first
voxel, `:fixed` draws one random origin at construction, and `:random` draws a new origin before every prox
evaluation. For the two shifted variants the grid wraps circularly, which stays a permutation of the voxels
only if every spatial extent is divisible by the corresponding block edge, so that is required.
"""
struct BlockNuclearNorm{R <: Real, N, RNG}
    λ::R
    block_size::NTuple{N, Int}
    spatial_size::NTuple{N, Int}
    num_frames::Int
    num_batch::Int
    threaded::Bool
    # The tiling depends only on the fields above, so it is computed once instead of per prox call.
    blocks::Vector{NTuple{N, UnitRange{Int}}}
    shift::Symbol
    # Origin of the tiling grid; mutable because `shift == :random` redraws it before every prox call.
    offset::Base.RefValue{NTuple{N, Int}}
    rng::RNG
end

function BlockNuclearNorm(
        λ::R, block_size::NTuple{N, Int}, spatial_size::NTuple{N, Int},
        num_frames::Int, num_batch::Int, threaded::Bool;
        shift::Symbol = :none, rng::RNG = Random.default_rng()
    ) where {R <: Real, N, RNG}
    @argcheck shift in (:none, :fixed, :random) "shift must be :none, :fixed or :random, got :$shift"
    if shift !== :none
        @argcheck all(spatial_size .% block_size .== 0) (
            "a shifted block grid wraps around the image, so it tiles the voxels exactly only if every " *
                "spatial extent is divisible by the block edge; got size $spatial_size with blocks $block_size"
        )
    end
    offset = shift === :none ? ntuple(_ -> 0, Val(N)) : _draw_offset(rng, block_size)
    return BlockNuclearNorm{R, N, RNG}(
        λ, block_size, spatial_size, num_frames, num_batch, threaded,
        _block_ranges(spatial_size, block_size), shift, Ref(offset), rng
    )
end

_draw_offset(rng, block_size::NTuple{N, Int}) where {N} = ntuple(d -> rand(rng, 0:(block_size[d] - 1)), Val(N))

ProximalCore.is_convex(::Type{<:BlockNuclearNorm}) = true
ProximalCore.is_positively_homogeneous(::Type{<:BlockNuclearNorm}) = true
ProximalCore.is_smooth(::Type{<:BlockNuclearNorm}) = false
ProximalCore.is_separable(::Type{<:BlockNuclearNorm}) = false

"""
	_block_ranges(spatial_size, block_size)

Return the tuple of index ranges of every block of the (possibly incomplete at the boundary) tiling of
`spatial_size` with blocks of size `block_size`.
"""
function _block_ranges(spatial_size::NTuple{N, Int}, block_size::NTuple{N, Int}) where {N}
    per_dim = ntuple(
        d -> [i:min(i + block_size[d] - 1, spatial_size[d]) for i in 1:block_size[d]:spatial_size[d]],
        Val(N)
    )
    return vec([ranges for ranges in Iterators.product(per_dim...)])
end

_reshaped(f::BlockNuclearNorm, x) = reshape(x, f.spatial_size..., f.num_frames, f.num_batch)

# Index tuple of one block under the current grid origin. Without a shift the block is a contiguous
# range per dimension; with one it wraps around the image and needs an explicit index vector.
function _block_indices(f::BlockNuclearNorm{<:Real, N}, ranges::NTuple{N, UnitRange{Int}}) where {N}
    offset = f.offset[]
    all(iszero, offset) && return ranges
    return ntuple(d -> [mod1(i + offset[d], f.spatial_size[d]) for i in ranges[d]], Val(N))
end

# Gather one block into a dense `(voxels × frames)` Casorati matrix. The block is a strided but
# non-contiguous view, so it has to be copied before the SVD anyway.
function _gather_block!(M, xr, idx::NTuple{N, Any}, batch) where {N}
    block = @view xr[idx..., :, batch]
    return copyto!(M, reshape(block, size(M)...))
end

function _scatter_block!(xr, M, idx::NTuple{N, Any}, batch) where {N}
    block = @view xr[idx..., :, batch]
    return copyto!(block, reshape(M, size(block)...))
end

function _block_matrix(f::BlockNuclearNorm, ::Type{T}, ranges) where {T}
    return Matrix{T}(undef, prod(length.(ranges)), f.num_frames)
end

function (f::BlockNuclearNorm)(x)
    _is_device(x) && return BatchedNuclearNorm(f.λ)(_to_block_stack(f, x))
    xr = _reshaped(f, x)
    R = real(eltype(x))
    value = R(0)
    for batch in 1:(f.num_batch), ranges in f.blocks
        idx = _block_indices(f, ranges)
        M = _block_matrix(f, eltype(x), ranges)
        _gather_block!(M, xr, idx, batch)
        value += sum(svdvals!(M))
    end
    return f.λ * value
end

function ProximalCore.prox!(y, f::BlockNuclearNorm, x, gamma)
    xr = _reshaped(f, x)
    yr = _reshaped(f, y)
    R = real(eltype(x))
    threshold = f.λ * gamma
    # A new grid origin per call is what makes the block artifacts of a single fixed tiling average out
    # over the iterations; it also makes the objective non-stationary (see `LocallyLowRank`).
    f.shift === :random && (f.offset[] = _draw_offset(f.rng, f.block_size))
    if _is_device(x)
        blocks = _to_block_stack(f, x)
        value = prox!(blocks, BatchedNuclearNorm(f.λ), blocks, gamma)
        _from_block_stack!(y, f, blocks)
        return value
    end
    nblocks = length(f.blocks)
    # One task per (block, batch slice) pair; every task writes into a disjoint part of `y`.
    partial = zeros(R, nblocks * f.num_batch)
    if f.threaded
        @budgeted_threads for k in eachindex(partial)
            partial[k] = _prox_block!(yr, xr, f, k, nblocks, threshold)
        end
    else
        for k in eachindex(partial)
            partial[k] = _prox_block!(yr, xr, f, k, nblocks, threshold)
        end
    end
    return f.λ * sum(partial)
end

function _prox_block!(yr, xr, f::BlockNuclearNorm, k::Int, nblocks::Int, threshold)
    ranges = f.blocks[mod1(k, nblocks)]
    batch = cld(k, nblocks)
    R = real(eltype(xr))
    idx = _block_indices(f, ranges)
    M = _block_matrix(f, eltype(xr), ranges)
    _gather_block!(M, xr, idx, batch)
    # Grants BLAS's threads for a large block, up to the budget `@budgeted_threads` gave this
    # task; see `with_serial_blas`.
    F = ProximalOperators.with_factorization_threads(() -> svd!(M), M)
    σ = max.(R(0), F.S .- threshold)
    lmul!(Diagonal(σ), F.Vt)
    mul!(M, F.U, F.Vt)
    _scatter_block!(yr, M, idx, batch)
    return sum(σ)
end

# Device path: every block's Casorati matrix at once, as the slices of one `m × frames × nblocks`
# array built by array operations only (a shift, a zero pad, a reshape and a permutation), so
# that one batched singular value operation handles them all. Zero rows leave a matrix's
# singular values and its singular value thresholding unchanged, so padding a partial edge block
# is exact.

_padded_spatial_size(f::BlockNuclearNorm) = cld.(f.spatial_size, f.block_size) .* f.block_size

function _to_block_stack(f::BlockNuclearNorm{<:Real, N}, x) where {N}
    xr = _reshaped(f, x)
    offset = f.offset[]
    all(iszero, offset) || (xr = circshift(xr, (map(-, offset)..., 0, 0)))
    padded = _padded_spatial_size(f)
    if padded != f.spatial_size
        xp = fill!(similar(xr, padded..., f.num_frames, f.num_batch), zero(eltype(xr)))
        copyto!(view(xp, map(Base.OneTo, f.spatial_size)..., :, :), xr)
        xr = xp
    end
    nb = padded .÷ f.block_size
    split = reshape(xr, _interleave(f.block_size, nb)..., f.num_frames, f.num_batch)
    stacked = permutedims(split, _block_stack_perm(Val(N)))
    return reshape(stacked, prod(f.block_size), f.num_frames, :)
end

function _from_block_stack!(y, f::BlockNuclearNorm{<:Real, N}, blocks) where {N}
    padded = _padded_spatial_size(f)
    nb = padded .÷ f.block_size
    stacked = reshape(blocks, f.block_size..., f.num_frames, nb..., f.num_batch)
    split = permutedims(stacked, invperm(_block_stack_perm(Val(N))))
    xr = reshape(split, padded..., f.num_frames, f.num_batch)
    if padded != f.spatial_size
        xr = xr[map(Base.OneTo, f.spatial_size)..., :, :]
    end
    offset = f.offset[]
    all(iszero, offset) || (xr = circshift(xr, (offset..., 0, 0)))
    copyto!(_reshaped(f, y), xr)
    return y
end

_interleave(a::NTuple{N, Int}, b::NTuple{N, Int}) where {N} = ntuple(i -> isodd(i) ? a[cld(i, 2)] : b[i ÷ 2], Val(2N))

# `(b₁, n₁, …, b_N, n_N, frames, batch)` to `(b₁, …, b_N, frames, n₁, …, n_N, batch)`.
_block_stack_perm(::Val{N}) where {N} =
    (ntuple(i -> 2i - 1, Val(N))..., 2N + 1, ntuple(i -> 2i, Val(N))..., 2N + 2)

"""
	LocallyLowRank(λ; block_size, time_dim=nothing, coil_dim=:coil, shift=:none, rng=Random.default_rng())
	LLR(λ; block_size, time_dim=nothing, coil_dim=:coil, shift=:none, rng=Random.default_rng())

Create a locally low-rank (LLR) regularization term with parameter `λ`. `LLR` is an exported alias:
BART, RegularizedLeastSquares.jl and the literature all use the acronym, and the two names are the
same type. The image is tiled into
non-overlapping spatial blocks, and the nuclear norm of the Casorati matrix of every block is penalized:
`λ ∑_b ‖𝓧_b‖_*`, where the rows of `𝓧_b` are the voxels of block `b` and its columns are the frames along
`time_dim`.

# Arguments
- `λ`: Regularization parameter, must be a scalar.
- `block_size`: Edge length of the blocks. Either an `Integer` (same length in every spatial dimension) or a
  tuple with one entry per spatial dimension (the dimensions before `time_dim`, not counting a coil axis).
  Typical values are 4-16.
- `time_dim`: (optional) Dimension holding the frames / contrasts. Can be an `Integer` (1-based index) or a
  `Symbol` (dimension name). If not provided, it will be inferred as the dimension named `:time` if `x` is a
  `NamedDimsArray`.
- `coil_dim`: (optional) Coil axis of an image reconstructed without sensitivity maps, which keeps one image
  per coil. Can be an `Integer` or a `Symbol`; the default `:coil` finds the dimension named `:coil` and is
  ignored when there is none. A coil axis before `time_dim` is never split into blocks: every block holds all
  coils, and their voxels are stacked into one Casorati matrix (see Notes). To get one independent matrix
  per coil instead, give a `block_size` tuple that includes an entry of `1` for the coil axis; a tuple with an
  entry for every dimension before `time_dim` is taken as given. A coil axis after `time_dim` is a batch
  dimension, so every coil is regularized separately.
- `shift`: (optional) Where the tiling grid starts. `:none` (default) anchors it at the first voxel, `:fixed`
  draws one random origin when the term is materialized, and `:random` draws a new origin before every
  proximal evaluation. See the notes on block artifacts below.
- `rng`: (optional) Random number generator used to draw the grid origin for `shift in (:fixed, :random)`.

# Notes
- LLR exploits that a *local* neighbourhood has far fewer independent temporal dynamics than the whole
  image, so it is a strictly stronger prior than the globally low-rank [`LowRank`](@ref) whenever the
  dynamics vary across the field of view. Introduced for dynamic MRI and parameter mapping by
  Trzasko & Manduca, *Local versus global low-rank promotion in dynamic MRI series reconstruction*,
  ISMRM 2011, and Zhang, Pauly & Levesque, *Accelerating parameter mapping with a locally low rank
  constraint*, Magn Reson Med 2015.
- The prox is exact: since the blocks tile the image without overlap, the block decomposition is a
  permutation of the voxels and the proximal operator decouples into one singular value thresholding per
  block. This stays true for a shifted grid, which wraps around the image and is therefore still a
  permutation — but only when every spatial extent is divisible by the block edge, which is checked.
- A single fixed grid can leave visible block artifacts at the block boundaries. The usual remedy in the
  literature is to shift the grid randomly between iterations (`shift=:random`), which averages the artifacts
  out. That makes the objective change from iteration to iteration, so it is no longer the same optimization
  problem at every step: use it only with `ISTA`, `FISTA` or `ADMM`. Line-search algorithms (`PANOC`,
  `PANOCplus`, `ZeroFPR`) compare objective values across iterations and can stall or fail on a moving
  objective. `shift=:fixed` keeps a single grid — the objective stays stationary and every algorithm remains
  valid — but merely moves the artifacts instead of removing them; it is mostly useful for averaging over
  independent runs.
- Coil images without sensitivity maps are the same object weighted by smooth coil sensitivities, so within
  one block they share their temporal dynamics. Stacking the coils into one Casorati matrix keeps its rank
  close to that of a single coil while multiplying the number of rows, which separates signal from noise
  better than thresholding every coil on its own. The singular values grow roughly with the square root of
  the number of coils, so `λ` tuned on single-coil data acts weaker on many coils.
- Cost is dominated by one SVD of a `(∏ block_size) × n_frames` matrix per block and iteration, so small
  blocks are cheaper but capture less spatial correlation.
"""
struct LocallyLowRank{T, B, D, C, RNG} <: Regularization
    λ::T
    block_size::B
    time_dim::D
    coil_dim::C
    shift::Symbol
    rng::RNG
    function LocallyLowRank(
            λ::T; block_size::B, time_dim::D = nothing, coil_dim::C = :coil,
            shift::Symbol = :none, rng::RNG = Random.default_rng()
        ) where {T, B, D, C, RNG}
        @argcheck λ isa Real "LocallyLowRank requires a scalar λ"
        @argcheck block_size isa Integer || block_size isa Tuple{Vararg{Integer}} "block_size must be an Integer or a tuple of Integers"
        @argcheck all(block_size .> 0) "block_size must be positive"
        @argcheck shift in (:none, :fixed, :random) "shift must be :none, :fixed or :random, got :$shift"
        _check_dim_spec(time_dim, "time_dim")
        _check_dim_spec(coil_dim, "coil_dim")
        return new{T, B, D, C, RNG}(λ, block_size, time_dim, coil_dim, shift, rng)
    end
end

const LLR = LocallyLowRank

get_operator(::LocallyLowRank, x::AbstractArray; threaded::Bool = true) = identity_operator(x)

function get_affected_dims(reg::LocallyLowRank, ::Nothing, image_dims)
    # Blocks couple all spatial dimensions up to (and including) the temporal one.
    return image_dims[1:get_time_dim(reg.time_dim, image_dims)]
end

# The nuclear norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
function scale_regularization(reg::LocallyLowRank, factor::Real)
    return LocallyLowRank(
        reg.λ * factor;
        block_size = reg.block_size, time_dim = reg.time_dim, coil_dim = reg.coil_dim,
        shift = reg.shift, rng = reg.rng
    )
end

function bind_dimensions(reg::LocallyLowRank, image_dims)
    return LocallyLowRank(
        reg.λ;
        block_size = reg.block_size,
        time_dim = get_time_dim(reg.time_dim, image_dims),
        coil_dim = _resolve_coil_dim(reg.coil_dim, image_dims),
        shift = reg.shift,
        rng = reg.rng,
    )
end

# Index of the coil axis, or `nothing` when the image has none. Unlike `time_dim` a coil axis is optional,
# so a name that is not found (or a name on an unnamed image) means "no coil axis" instead of an error.
function _resolve_coil_dim(coil_dim, image_dims)
    coil_dim isa Symbol || return coil_dim
    image_dims[1] isa Symbol || return nothing
    return findfirst(==(coil_dim), image_dims)
end

# Block size over the `N` dimensions before the temporal one. A coil axis among them (`coil_dim`) is
# covered by a single block, so every block stacks all coils into one Casorati matrix; the requested size
# then names only the other dimensions. A tuple with an entry for every dimension is taken as given (an
# entry of 1 on the coil axis regularizes each coil on its own). A shifted grid stays exact on the coil
# axis: the block spans the whole axis, so the circular shift only reorders the rows of each matrix.
function _llr_block_size(requested, spatial_size::NTuple{N, Int}, coil_dim = nothing) where {N}
    joint = coil_dim isa Integer && coil_dim <= N
    if requested isa Integer
        block_size = ntuple(d -> joint && d == coil_dim ? spatial_size[d] : Int(requested), Val(N))
    elseif joint && length(requested) == N - 1
        block_size = ntuple(d -> d == coil_dim ? spatial_size[d] : Int(requested[d < coil_dim ? d : d - 1]), Val(N))
    else
        block_size = Int.(requested)
    end
    expected = joint ? "$(N - 1) (or $N with the coil axis)" : "$N"
    @argcheck length(block_size) == N "block_size must have one entry per spatial dimension ($expected), got $(length(requested))"
    @argcheck all(block_size .<= spatial_size) "block_size $(block_size) exceeds the spatial image size $(spatial_size) (the dimensions before time_dim)"
    return NTuple{N, Int}(block_size)
end

function materialize(reg::LocallyLowRank, x::Variable{T}; threaded::Bool) where {T}
    x_val = ~x
    dims = dims_of(x_val)
    time_dim = get_time_dim(reg.time_dim, dims)
    @argcheck time_dim > 1 "LocallyLowRank needs at least one spatial dimension before the temporal one"
    spatial_size = NTuple{time_dim - 1, Int}(size(x_val)[1:(time_dim - 1)])
    block_size = _llr_block_size(reg.block_size, spatial_size, _resolve_coil_dim(reg.coil_dim, dims))
    num_batch = prod(size(x_val)[(time_dim + 1):end]; init = 1)
    f = BlockNuclearNorm(
        real(T)(reg.λ), block_size, spatial_size, size(x_val, time_dim), num_batch, threaded;
        shift = reg.shift, rng = reg.rng
    )
    op = get_operator(reg, x_val; threaded)
    repr = @sprintf "%g ⋅ ∑_b ‖𝓧_b(%s)‖_*" real(T)(reg.λ) get_name(x)
    return StructuredOptimization.Term(1, f, op * x, repr)
end

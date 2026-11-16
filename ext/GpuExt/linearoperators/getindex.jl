function _to_gpu_indices(ref_array::AbstractGPUArray, cpu_idx::AbstractVector{<:Integer})
    ArrayT = Base.typename(typeof(ref_array)).wrapper
    return ArrayT(Vector{Int}(cpu_idx))
end

function _to_gpu_indices(array_type::Type, cpu_idx::AbstractVector{<:Integer})
    ArrayT = Base.typename(array_type).wrapper
    return ArrayT(Vector{Int}(cpu_idx))
end

function _mask_to_linear_indices(mask::AbstractArray{Bool})
    return findall(vec(mask))
end

function _mask_to_linear_indices(mask::AbstractGPUArray{Bool})
    return findall(vec(Array(mask)))
end

function AbstractOperators._prepare_getindex_intvec(
        idx::Vector{Int}, array_type::Type{<:AbstractGPUArray}
    )
    return _to_gpu_indices(array_type, idx)
end

function AbstractOperators._prepare_getindex_boolmask(
        mask::AbstractArray{Bool}, array_type::Type{<:AbstractGPUArray}
    )
    return _to_gpu_indices(array_type, _mask_to_linear_indices(mask))
end

# One component of an index tuple on the device: a Bool mask as the positions it selects (Int for
# one dimension, `CartesianIndex` for several, as Base indexes with a mask), an index array as is;
# scalars, ranges and `Colon` need no storage and stay.
_tuple_index_to_gpu(array_type, i) = i
_tuple_index_to_gpu(array_type, i::AbstractVector{Bool}) = _to_gpu_indices(array_type, findall(Array(i)))
function _tuple_index_to_gpu(array_type, i::AbstractArray{Bool})
    ArrayT = Base.typename(array_type).wrapper
    return ArrayT(findall(Array(i)))
end
_tuple_index_to_gpu(array_type, i::AbstractVector{<:Integer}) = _to_gpu_indices(array_type, i)
_tuple_index_to_gpu(array_type, i::AbstractRange{<:Integer}) = i
function _tuple_index_to_gpu(array_type, i::AbstractVector{<:CartesianIndex})
    ArrayT = Base.typename(array_type).wrapper
    return ArrayT(Vector(i))
end

# A host index array in the tuple would be copied to the device, and its bounds checked there with
# a reduction read back to the host, on every application: per call one upload and one
# synchronisation, which on a per-coil, per-frame batch of small `GetIndex`es dominated the
# operator. Moved once here, and the applications below skip the bounds check that construction
# has done.
function AbstractOperators._prepare_getindex_tuple(idx::Tuple, array_type::Type{<:AbstractGPUArray})
    return map(i -> _tuple_index_to_gpu(array_type, i), idx)
end

function GetIndex(x::AbstractGPUArray, idx::AbstractVector{Int})
    dim_in = size(x)
    dim_out = AbstractOperators.get_dim_out(dim_in, idx)
    if dim_out == dim_in
        return Eye(eltype(x), dim_in; array_type = typeof(x))
    end
    S = AbstractOperators._array_wrapper(x){eltype(x)}
    gpu_idx = _to_gpu_indices(x, idx)
    return GetIndex(eltype(x), S, dim_out, dim_in, gpu_idx)
end

function GetIndex(x::AbstractGPUArray, mask::AbstractArray{Bool})
    dim_in = size(x)
    dim_out = AbstractOperators.get_dim_out(dim_in, mask)
    if dim_out[1] == prod(dim_in)
        return reshape(Eye(eltype(x), dim_in; array_type = typeof(x)), dim_out)
    end
    S = AbstractOperators._array_wrapper(x){eltype(x)}
    gpu_idx = _to_gpu_indices(x, _mask_to_linear_indices(mask))
    return GetIndex(eltype(x), S, dim_out, dim_in, gpu_idx)
end

function mul!(
        y::AbstractGPUArray, L::GetIndex{I}, b::AbstractGPUArray
    ) where {K, I <: NTuple{K, Any}}
    check(y, L, b)
    y .= @inbounds view(b, L.idx...)
    return y
end

function mul!(
        y::AbstractGPUArray, Lc::AdjointOperator{<:GetIndex{I}}, b::AbstractGPUArray
    ) where {K, I <: NTuple{K, Any}}
    check(y, Lc, b)
    fill!(y, zero(eltype(y)))
    @inbounds view(y, Lc.A.idx...) .= b
    return y
end

function mul!(
        y::AbstractGPUArray, L::GetIndex{I}, b::AbstractGPUArray, α::Number, β::Number
    ) where {K, I <: NTuple{K, Any}}
    check(y, L, b)
    return AbstractOperators._store!(y, view(b, L.idx...), α, β)
end

# The additive adjoint as one broadcast into the selected samples, which on a device avoids
# gathering them into a copy and checking its indices with a reduction read back to the host.
function mul!(
        y::AbstractGPUArray, Lc::AdjointOperator{<:GetIndex{I}}, b::AbstractGPUArray, α::Number, β::Number
    ) where {K, I <: NTuple{K, Any}}
    check(y, Lc, b)
    AbstractOperators._scale_output!(y, β)
    dst = view(y, Lc.A.idx...)
    a = AbstractOperators._coefficient(y, α)
    dst .+= a .* b
    return y
end

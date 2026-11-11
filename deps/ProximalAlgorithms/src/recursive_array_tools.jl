using RecursiveArrayTools: ArrayPartition

# The fork's `ProximalAlgorithmsRecursiveArrayToolsExt`, inlined: a package included as a submodule
# cannot load an extension. The reductions over an iterate go part by part, so that each part takes
# its own method: the loop for `Array` storage, a lazy broadcast for a device array. A broadcast over
# the partition itself indexes the elements of every part one by one, which a device array refuses.
_pogm_restart_sum(w::ArrayPartition, z::ArrayPartition, x::ArrayPartition) =
    sum(map(_pogm_restart_sum, w.x, z.x, x.x))

# The six squared norms of the Lipschitz safeguard, added over the parts.
function _safeguard_sums_in(
        ::Type{S}, x::ArrayPartition, x_prev::ArrayPartition, grad::ArrayPartition, grad_prev::ArrayPartition,
    ) where {S}
    parts = map((u...) -> _safeguard_sums_in(S, u...), x.x, x_prev.x, grad.x, grad_prev.x)
    return reduce((a, b) -> map(+, a, b), parts)
end

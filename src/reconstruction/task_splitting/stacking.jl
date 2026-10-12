function robust_global_scale(scales)
    nonzero = filter(!iszero, scales)
    return isempty(nonzero) ? one(eltype(scales)) : median(nonzero)
end

function safe_scale_ratio(scale, global_scale)
    # Guard against a slice whose own scale estimate is zero (or negligible relative to the
    # rest of the slices, e.g. an empty/noise-only slice): shrinking λ towards zero there would
    # leave that slice's noise essentially unregularized, so fall back to no correction instead.
    ratio = scale / global_scale
    return abs(ratio) < 1.0e-6 ? one(ratio) : ratio
end

# `results` holds slices of whatever the per-slice reconstruction returned, so the element type is
# abstract; dispatch on one element rather than testing its type here.
function stack_image_slices(results, plan, threaded::Val)
    return stack_slices_like(first(results), results, plan, threaded)
end

function stack_slices_like(::AbstractArray, results, plan, threaded::Val)
    return stack_plain_image_slices(results, plan, threaded)
end

function stack_slices_like(::ReconImage, results, plan, threaded::Val)
    return stack_split_image_slices(results, plan, threaded)
end

function stack_plain_image_slices(results, plan, ::Val{false})
    full_image = similar(unname(results[1]), plan.output_size)
    for (output_slice, result) in
        zip(eachslice(full_image; dims = plan.variable_batch_dims), results)
        output_slice .= unname(result)
    end
    return full_image
end

function stack_plain_image_slices(results, plan, ::Val{true})
    full_image = similar(unname(results[1]), plan.output_size)
    extended_results = collect(
        zip(eachslice(full_image; dims = plan.variable_batch_dims), results)
    )
    @threads for (output_slice, result) in extended_results
        output_slice .= unname(result)
    end
    return full_image
end

function stack_split_image_slices(results, plan, threaded::Val)
    summed = stack_plain_image_slices(map(total_image, results), plan, threaded)
    names = keys(components(first(results)))
    comps = NamedTuple{names}(
        Tuple(
            stack_plain_image_slices(map(r -> components(r)[name], results), plan, threaded)
                for name in names
        )
    )
    return ReconImage(summed, Header(), comps, _spatial_ndims(first(results)))
end

function maybe_rescale_results!(results, scales, config)
    return if !config.disable_inverse_scale_output
        median_scale = median(scales)
        @threads for i in eachindex(results)
            _rescale_result!(results[i], median_scale)
        end
        median_scale != 1 &&
            log_message(config.verbosity, "Rescaled output by median scale factor $median_scale")
    end
end

_rescale_result!(x::AbstractArray, factor) = (x .*= factor)
_rescale_result!(x::ReconImage, factor) = rescale!(x, factor)

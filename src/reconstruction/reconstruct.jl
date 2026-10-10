"""
	reconstruct(
		acq_data::AcquisitionInfo,
		[method::ReconstructionMethod = DirectReconstruction()];
		[x₀], kwargs...)

Performs MRI reconstruction from k-space data using the specified reconstruction method.

# Arguments
- `acq_data::AcquisitionInfo`: The acquisition information containing k-space data, sensitivity maps, and other parameters.
- `method::ReconstructionMethod = DirectReconstruction()`: The reconstruction method (e.g. `DirectReconstruction()`, `IterativeReconstruction(...)`).

# Keyword arguments
- `x₀::Union{Nothing,AbstractArray,Tuple,NamedTuple}=nothing`: Optional initial guess for the image (default is 𝒜' * y).
- `config::ReconstructionConfig`: an existing [`ReconstructionConfig`](@ref) to extend; the keywords below override its fields.
- `scaling::Scaling = QuantileScaling()`: scaling applied to operators/data (see [`ReconstructionConfig`](@ref) for the others)
- `verbosity::Verbosity = Silent()`: output mode — [`Silent`](@ref), [`ProgressBar`](@ref) or [`Verbose`](@ref)
- `threaded::Bool = (Threads.nthreads() > 1)`: enable threaded execution when available
- `task_executor::Union{Nothing,ReconstructionExecutor} = nothing`: override executor for task splitting
- `disable_inverse_scale_output::Bool = false`: skip rescaling the final output
- `disable_task_splitting::Union{Nothing,Bool} = nothing`: disable automatic task splitting;
  `nothing` splits on the host and follows `DEVICE_DISABLES_TASK_SPLITTING` on a device

# Where it runs
The reconstruction runs where `acq_data`'s k-space lives. Move an acquisition to a GPU with
`Adapt.adapt(CuArray, acq)` (or any other device array type); the result is then a device array
too. `x₀`, if given, must be in the same memory as the k-space. See the "GPU reconstruction"
page of the manual for what runs on the device and what is staged through the host.

Iteration control is *not* accepted here: `maxit`, `reltol` and `algorithm` are properties of the
method and are passed to its constructor, e.g.
`reconstruct(acq, IterativeReconstruction(reg; maxit = 50, reltol = 1e-6))` or
`reconstruct(acq, POCS(; maxit = 20))`. Passing them to `reconstruct` throws.

# Returns
- A [`ReconImage`](@ref): the reconstructed image (a `NamedDimsArray` inside if the k-space was
  named) with a copy of the acquisition's [`header`](@ref).
"""
function reconstruct(
        acq_data::AcquisitionInfo,
        method::ReconstructionMethod = DirectReconstruction();
        x₀::Union{Nothing, AbstractArray, Tuple, NamedTuple} = nothing,
        kwargs...,
    )
    _check_x₀_storage(x₀, acq_data)
    # Resolved against the host copy there: no slice of a host-only method shares the device.
    if _is_device(acq_data) && _runs_on_host(method)
        host_x = reconstruct(Adapt.adapt(Array, acq_data), method; x₀ = _adapt_any(Array, x₀), kwargs...)
        return _to_storage_of(acq_data, host_x)
    end
    config = resolve_config(construct_config(kwargs), acq_data, method)
    t_start = time()
    method = lower(method, acq_data)
    check_applicable(method, acq_data)
    # What the measured plans learned is saved once per reconstruction, not only at exit, so a
    # session that is killed does not plan them again next time.
    x = try
        _reconstruct_dispatch(acq_data, method, x₀, config)
    finally
        _save_fftw_wisdom()
    end
    t_end = time()
    log_message(config.verbosity, "Total time: ", format_time(t_end - t_start))
    return _with_header(x, acq_data)
end

# The image `reconstruct` returns: the solution with a copy of the acquisition's header, `spacing`
# derived from `fov` when only that is known.
function _with_header(x, acq_data)
    nd = length(acq_data.image_size)
    h = _derive_spacing!(copy(header(acq_data)), size(x)[1:nd])
    x isa ReconImage || return ReconImage(x, h, nothing, nd)
    return ReconImage(parent(x), h, components(x), nd)
end

"""
    _runs_on_host(method::ReconstructionMethod) -> Bool

Whether `method` has no device implementation, so that a device acquisition is reconstructed on
a host copy and the image moved back (GRAPPA's and SPIRiT's kernels are scalar convolutions).
"""
_runs_on_host(::ReconstructionMethod) = false

# An initial guess must live where the k-space does: the solver's iterates start as copies of it.
_check_x₀_storage(x₀, acq_data) = _check_same_storage(acq_data, x₀, "x₀")
_check_x₀_storage(x₀::Union{Tuple, NamedTuple}, acq_data) = foreach(x -> _check_x₀_storage(x, acq_data), x₀)

function _reconstruct_dispatch(acq_data, method::ReconstructionMethod, x₀, config)
    @argcheck isnothing(x₀) || x₀ isa AbstractArray "x₀ must be a plain array unless reconstructing with `Component`s."
    return _reconstruct_dispatch_plain(acq_data, method, x₀, config)
end

function _reconstruct_dispatch(acq_data, method::IterativeReconstruction, x₀, config)
    if method.regularization isa Tuple{Component, Vararg{Component}}
        check_components(method.regularization)
        return _reconstruct_dispatch_components(acq_data, method, x₀, config)
    else
        @argcheck isnothing(x₀) || x₀ isa AbstractArray "x₀ must be a plain array unless reconstructing with `Component`s."
        return _reconstruct_dispatch_plain(acq_data, method, x₀, config)
    end
end

function _reconstruct_dispatch_plain(acq_data, method::ReconstructionMethod, x₀, config)
    task_splitting_plan = get_task_splitting_plan(acq_data, method, config)
    x = if isnothing(task_splitting_plan)
        # Unsplit: this is where the one progress bar per `reconstruct` call is opened.
        # `progress_total` decides whether it is a determinate bar over the method's own loop or
        # the indeterminate stage indicator driven by the `@step` brackets.
        with_progress(config.verbosity, progress_total(method, acq_data)) do verbosity
            # An unsplit problem has no slices to spread, so `config.threaded` is passed through
            # as the caller set it and every operator decides for itself whether its own kernel
            # is worth threading at the size it is handed.
            conf = ReconstructionConfig(config; verbosity)
            reconstruction_result = nothing
            @conditionally_enable_threading conf.threaded begin
                reconstruction_result = _reconstruct(acq_data, method, x₀, conf)
            end
            first(reconstruction_result)
        end
    else
        if !isnothing(x₀)
            @argcheck size(x₀) == task_splitting_plan.variable_size "Size of x₀ ($(size(x₀))) must match the variable size ($(task_splitting_plan.variable_size))"
        end
        result = if method isa DirectMethod
            # Direct reconstruction needs no scaling; keep slices identical to the
            # unsplit result instead of normalizing each slice separately. A *new*
            # binding, not a reassignment of `config`: rebinding it would box the variable that
            # the `with_progress` closure above captures.
            unscaled_config = ReconstructionConfig(config; scaling = NoScaling())
            # Every slice builds an encoding operator of the same shape and drops it right after
            # its one adjoint, so the slices share one pool: each build takes the Compose buffers
            # an earlier slice returned, and the FFT plans an earlier slice planned.
            pool = AbstractOperators.OperatorPool()
            execute(task_splitting_plan, acq_data, unscaled_config) do idx, local_acq, local_conf
                local_x₀ = isnothing(x₀) ? nothing : get_x₀_slice(x₀, task_splitting_plan, idx)
                AbstractOperators.with_operator_pool(pool) do
                    𝒜 = build_encoding_operator(
                        local_acq, method;
                        threaded = local_conf.threaded,
                        fast_planning = _fast_planning(method, local_acq, local_conf),
                    )
                    slice_result = _reconstruct(local_acq, method, local_x₀, local_conf; 𝒜)
                    AbstractOperators.recycle!(pool, 𝒜)
                    _release_device_plans!(𝒜, _storage_template(local_acq))
                    slice_result
                end
            end
        else
            # Each slice's regularization strength is scale-dependent, so each slice is
            # first solved with its own scale to size λ correctly (via scale_regularization),
            # then the actual solve and the final image use one shared scale across all
            # slices so the output intensities are consistent slice-to-slice.
            execute_regularized(task_splitting_plan, acq_data, config, method, x₀)
        end
        if _has_dimnames(acq_data.kspace_data)
            result = NamedDimsArray{output_dims(method, acq_data)}(unname(result))
        end
        result
    end
    return x
end

function _reconstruct(
        acq_data, method::ReconstructionMethod, x₀, config;
        scale_override = nothing, 𝒜 = nothing, prior = nothing, density_weights = _density_weights,
    )
    fast_planning = _fast_planning(method, acq_data, config)
    built_here = isnothing(𝒜)
    if built_here
        @step "Constructing encoding operator" config begin
            𝒜 = build_encoding_operator(
                acq_data, method; threaded = config.threaded, fast_planning
            )
        end
    end

    x̂, scale, direct_prior = _direct_reconstruct(𝒜, acq_data, x₀, method, config; scale_override)
    prior = something(prior, direct_prior)

    if method isa DirectMethod
        if scale != 1 && config.disable_inverse_scale_output
            @step "Scaling image" config begin
                x̂ ./= scale
            end
        end
    elseif method isa IterativeReconstruction
        bound_regs = bind_dimensions(method.regularization, get_image_dims(acq_data))
        preconditioner = _chambolle_pock_preconditioner(method, acq_data; density_weights)
        build = (𝒜, y; x₀) -> build_model_with_variables(
            𝒜, y, bound_regs;
            threaded = config.threaded, x₀,
            fidelity = method.fidelity, preconditioner,
        )
        # The same two post-processing steps the final image goes through below, so that an
        # `on_iteration` callback sees intermediate iterates in the units, shape and dimension
        # names of the value this function returns.
        present = x -> _present_image(x, method, acq_data, config)
        x̂ = _iterative_reconstruct_core(
            𝒜, acq_data, x̂, scale, method, config; build, present, prior, preconditioner,
        )
        x̂ = _present_image(x̂, method, acq_data, config)
    end

    built_here && _release_device_plans!(𝒜, _storage_template(acq_data))
    return x̂, scale
end

# Signal model + dimension names: the last two steps between a solved variable and the image the
# caller gets. Factored out because the `on_iteration` callback has to apply exactly the same two
# to every intermediate iterate.
function _present_image(x, method::IterativeReconstruction, acq_data, config)
    x = apply_signal_model(method.signal_model, x, acq_data; threaded = config.threaded)
    if _has_dimnames(acq_data.kspace_data) && !(x isa NamedDimsArray)
        x = NamedDimsArray{output_dims(method, acq_data)}(x)
    end
    return x
end

function _reconstruct_dispatch_components(acq_data, method::IterativeReconstruction, x₀, config)
    task_splitting_plan = get_task_splitting_plan(acq_data, method, config)
    components = method.regularization
    img = if isnothing(task_splitting_plan)
        # The task-splitting branch below validates x₀ against the plan's image size; this branch has
        # no plan, so it validates against the acquisition's own image size. Both must check, or a
        # mistyped component name is only caught when the task happens to be split.
        if !isnothing(x₀)
            check_x₀_components_size(x₀, components, get_image_size(acq_data))
        end
        with_progress(config.verbosity, progress_total(method, acq_data)) do verbosity
            # As in `_reconstruct_dispatch_plain`: nothing to spread, so the operators decide.
            conf = ReconstructionConfig(config; verbosity)
            result = nothing
            @conditionally_enable_threading conf.threaded begin
                result = _reconstruct_components(acq_data, method, x₀, conf)
            end
            first(result)
        end
    else
        if !isnothing(x₀)
            check_x₀_components_size(x₀, components, task_splitting_plan.variable_size)
        end
        execute_regularized_components(task_splitting_plan, acq_data, config, method, x₀)
    end
    if _has_dimnames(acq_data.kspace_data) && !(total_image(img) isa NamedDimsArray)
        img_dimnames = output_dims(method, acq_data)
        img = ReconImage(
            NamedDimsArray{img_dimnames}(unname(total_image(img))), Header(),
            map(c -> NamedDimsArray{img_dimnames}(unname(c)), getfield(img, :components)), _spatial_ndims(img),
        )
    end
    return img
end

function _reconstruct_components(
        acq_data, method::IterativeReconstruction, x₀, config;
        scale_override = nothing, x₀s = nothing, 𝒜 = nothing, prior = nothing,
    )
    components = bind_dimensions(method.regularization, get_image_dims(acq_data))
    built_here = isnothing(𝒜)
    if built_here
        @step "Constructing encoding operator" config begin
            𝒜 = build_encoding_operator(
                acq_data, method; threaded = config.threaded, fast_planning = _fast_planning(method, acq_data, config)
            )
        end
    end
    # `x₀s` lets a caller that has already formed the per-component initial guesses skip the adjoint
    # that would produce them. The task-splitting path computes them in its first phase to derive the
    # per-slice scales, and without this would recompute 𝒜'y per slice only to discard it.
    scale = if isnothing(x₀s)
        x̂, s, direct_prior = _direct_reconstruct_components(𝒜, acq_data, method, config; scale_override)
        x₀s = get_component_x0s(components, x̂, x₀)
        prior = something(prior, direct_prior)
        s
    else
        @argcheck !isnothing(scale_override) "scale_override is required when x₀s is supplied."
        scale_override
    end
    build = (𝒜, y; x₀) -> build_model(
        𝒜, y, components;
        threaded = config.threaded, x₀s = x₀,
        fidelity = method.fidelity,
    )
    names = map(c -> c.name, components)
    present = xs -> _present_components(xs, names, method, acq_data)
    xs = _iterative_reconstruct_core(𝒜, acq_data, x₀s, scale, method, config; build, present, prior = something(prior, _NO_PRIOR))
    built_here && _release_device_plans!(𝒜, _storage_template(acq_data))
    return _present_components(xs, names, method, acq_data), scale
end

# The component counterpart of `_present_image`: sum the per-component iterates into the total
# image and name both. An `on_iteration` callback on this path receives a `ReconImage` holding the
# components, as `reconstruct` returns, without the acquisition's header.
function _present_components(xs, names, method::IterativeReconstruction, acq_data)
    xs = _component_parts(xs)
    total_x = broadcast(+, xs...)
    if _has_dimnames(acq_data.kspace_data)
        img_dimnames = output_dims(method, acq_data)
        total_x = NamedDimsArray{img_dimnames}(total_x)
        xs = map(x -> NamedDimsArray{img_dimnames}(x), xs)
    end
    return ReconImage(total_x, Header(), NamedTuple{names}(xs), length(acq_data.image_size))
end

# `_extract_solution` hands back a `Tuple` of variables, but a solver *iterate* on the component
# path is the `ArrayPartition` the multi-variable problem is solved over. Both name the same
# per-component arrays, so normalize to a tuple before assembling the image.
_component_parts(xs::Tuple) = xs
_component_parts(xs::ArrayPartition) = xs.x

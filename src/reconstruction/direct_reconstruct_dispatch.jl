function _direct_reconstruct_components(𝒜, acq_data, method::ReconstructionMethod, config; scale_override = nothing)
    @step "Getting initial estimate" config begin
        x̂ = 𝒜' * _measurement(acq_data.kspace_data)
    end
    scale = _resolve_scale(𝒜, x̂, acq_data, method, config, scale_override)
    # Computed from the pre-rescale `x̂`, so `scale` above stays exactly as before.
    x̂s, L, curvature = _scale_default_warm_start(𝒜, x̂, _adjoint_measurement(acq_data), method, config)
    return x̂s, scale, _warm_start_prior(L, curvature)
end

# The scale is either imposed by the caller (task splitting uses one shared scale for every slice),
# derived from the direct estimate, or absent; a zero estimate would blow up the scaled problem, so it
# falls back to no scaling. `𝒜` is the operator `x̂` was formed with.
function _resolve_scale(𝒜, x̂, acq_data, method, config, scale_override)
    if !isnothing(scale_override)
        scale = scale_override
        log_message(config.verbosity, @sprintf("Using scaling factor: %g", scale))
    elseif config.scaling != NoScaling()
        @step "Computing scaling factor" config begin
            scale = get_scale(config.scaling, acq_data, _scale_input(𝒜, x̂, acq_data, method, config)...)
        end
        if scale == 0
            log_message(config.verbosity, "Warning: Computed scale is zero, defaulting to scale=1.0")
            scale = 1
        end
        log_message(config.verbosity, @sprintf("Using scaling factor: %g", scale))
    else
        scale = 1
    end
    return real(eltype(x̂))(scale)
end

# The image the scale is read from, and the operator that formed it: the image-only encoding's
# adjoint when a signal model sits between the variable and the image, else `x̂` and `𝒜` themselves.
function _scale_input(𝒜, x̂, acq_data, method, config)
    method isa IterativeReconstruction && method.signal_model !== nothing || return x̂, 𝒜
    E = get_encoding_operator(acq_data; threaded = config.threaded)
    return E' * _measurement(acq_data.kspace_data), E
end

"""
    _direct_coil_dim(acq::CartesianAcquisitionInfo)

Integer position of the coil axis in `acq.kspace_data`, or `0` when there is none. Resolved from
dimension names when the k-space is a `NamedDimsArray`, else assumed to immediately follow the
sampled k-space dimensions (`3` for a fully sampled 2D grid, `4` for 3D, one fewer for each pair of
axes a subsampling mask joins) -- unlike `_pf_coil_dim`, which hardcodes `3` and is only ever used by
the 2D-only partial-Fourier methods. [`_direct_image_coil_dim`](@ref) translates it to a position
among the image dimensions.
"""
function _direct_coil_dim(acq::CartesianAcquisitionInfo)
    if _has_dimnames(acq.kspace_data)
        idx = findfirst(==(:coil), dimnames(acq.kspace_data))
        return isnothing(idx) ? 0 : Int(idx)
    end
    sample_dims = _get_sample_dims_count(acq)
    return ndims(acq.kspace_data) > sample_dims ? sample_dims + 1 : 0
end

"""
    lower(method::DirectReconstruction, acq::AcquisitionInfo)

Resolve the default (`nothing`) coil combination against the acquisition. Without sensitivity maps
the coil axis is not part of the signal model at all: it is a batch dimension, and the
reconstruction is one independent image per channel, so `NoCoilCombination` is what actually
happens and what the method should say it does. With maps, the default is the SNR-optimal
`AdjointSensitivity`.
"""
function lower(method::DirectReconstruction{Nothing}, acq::AcquisitionInfo)
    combination = if isnothing(acq.sensitivity_maps) && _direct_coil_dim(acq) != 0
        NoCoilCombination()
    else
        AdjointSensitivity()
    end
    return DirectReconstruction(combination)
end

"""
    _direct_combines_coils(method::DirectReconstruction, acq::AcquisitionInfo)

Whether this (already lowered) `DirectReconstruction` reduces a coil axis that the acquisition
still carries in its *image* dimensions -- that is, a maps-less acquisition combined by
`RootSumSquares`. With sensitivity maps the coil axis is consumed by the encoding operator and is
not an image dimension to begin with, and `NoCoilCombination` reduces nothing.

It is the one case where the reconstructed image has fewer dimensions than `get_image_dims(acq)`
says, which is why `variable_size`, `output_dims` and task splitting all have to ask.
"""
function _direct_combines_coils(method::DirectReconstruction, acq::AcquisitionInfo)
    return isnothing(acq.sensitivity_maps) && _direct_coil_dim(acq) != 0 &&
        !(method.coil_combination isa Union{Nothing, NoCoilCombination})
end
_direct_combines_coils(::ReconstructionMethod, ::AcquisitionInfo) = false

"""
    _direct_image_coil_dim(acq::AcquisitionInfo)

Position of the coil axis among the *image* dimensions of a maps-less acquisition. It differs from
`_direct_coil_dim`, which is a k-space position, by the same offset task splitting uses to map
image dimensions onto k-space dimensions: a subsampled or partitioned k-space has fewer Fourier
dimensions than the image it encodes.
"""
function _direct_image_coil_dim(acq::AcquisitionInfo)
    c_dim = _direct_coil_dim(acq)
    c_dim == 0 && return 0
    return c_dim + length(get_fourier_image_dims(acq)) - length(get_fourier_kspace_dims(acq))
end

function variable_size(method::DirectReconstruction, acq::AcquisitionInfo)
    sz = get_image_size(acq)
    _direct_combines_coils(method, acq) || return sz
    c = _direct_image_coil_dim(acq)
    return tuple(sz[1:(c - 1)]..., sz[(c + 1):end]...)
end

function output_dims(method::DirectReconstruction, acq::AcquisitionInfo)
    dims = get_image_dims(acq)
    _direct_combines_coils(method, acq) || return dims
    c = _direct_image_coil_dim(acq)
    return tuple(dims[1:(c - 1)]..., dims[(c + 1):end]...)
end

variable_dims(method::DirectReconstruction, acq::AcquisitionInfo) = output_dims(method, acq)

"""
    check_applicable(method::DirectReconstruction, acq::AcquisitionInfo)

An *explicit* `AdjointSensitivity` on an acquisition without sensitivity maps is an error: that
combination is defined by the maps, and without them the coil axis is a batch dimension (see
`lower`), so the channels are never seen together.

`RootSumSquares` is not affected. It is defined without maps -- it is the maps-free reference
every sensitivity-based reconstruction is compared against -- and `_direct_reconstruct_coil_combined`
builds the per-coil images from a sensitivity-free encoding operator anyway, so naming it on a
maps-less acquisition is a request the method can honour exactly.
"""
function check_applicable(method::DirectReconstruction, acq::AcquisitionInfo)
    if isnothing(acq.sensitivity_maps) && _direct_coil_dim(acq) != 0 &&
            method.coil_combination isa AdjointSensitivity
        throw(
            ArgumentError(
                "AdjointSensitivity coil combination needs sensitivity maps, and this acquisition carries none -- " *
                    "its coil axis is a batch dimension, reconstructed one channel at a time. Drop the argument (or " *
                    "pass NoCoilCombination()) to get the per-channel images, pass RootSumSquares() for the maps-free " *
                    "combination, or attach sensitivity maps (see estimate_sensitivities)."
            )
        )
    end
    return nothing
end

"""
    _direct_reconstruct_coil_combined(acq_data, method::DirectReconstruction, 𝒜)

`DirectReconstruction`'s own coil combination: `𝒜' * kspace_data` bakes in `AdjointSensitivity`
combination whenever sensitivity maps are present (via `_compose_with_sensitivity`) and cannot
express `RootSumSquares` or `NoCoilCombination`, so those are dispatched explicitly here, the same
way the shared `_kspace_to_image` helper (used by GRAPPA/SPIRiT/`KSpaceToImage`) does -- but from a
freshly-built sensitivity-free encoding operator rather than assuming `kspace_data` is already a
complete (zero-filled) grid, since `DirectReconstruction` also runs on subsampled data. The
non-Cartesian method below does the same from the gridding adjoint.

`AdjointSensitivity` is `𝒜'` itself and reuses the operator it is handed. Rebuilding a
sensitivity-free operator for it gives the same image bit for bit, but costs a second operator
build per slice. In a task-split reconstruction (12 slices, 30 cine frames) those builds made up
half of all allocations. At 8-16 threads the slices also queued on FFTW's global planner lock and
spent 30-70 % of the time in GC. Measured on the torso cine adjoint (EPYC 7763), reusing `𝒜'`
took 47.6 → 27.4 ms at 1 thread and 55.0 → 11.6 ms at 16.
"""
function _direct_reconstruct_coil_combined(acq_data::CartesianAcquisitionInfo, method::DirectReconstruction, 𝒜)
    smaps = acq_data.sensitivity_maps
    if method.coil_combination isa AdjointSensitivity
        # With maps, `𝒜` composes the sensitivity operator (`_compose_with_sensitivity`), so its
        # adjoint is exactly this combination. Without them, `check_applicable` has already
        # rejected any acquisition with a coil axis; what reaches here is single-channel data,
        # where the combination is the bare adjoint.
        return 𝒜' * _measurement(acq_data.kspace_data)
    end
    if _direct_coil_dim(acq_data) == 0
        # No coil axis at all: nothing for any combination choice to do.
        return 𝒜' * _measurement(acq_data.kspace_data)
    end
    # The coil images below are image-shaped, so the coil axis is addressed by its image position:
    # a subsampled k-space (`(:kx, :kyz, :coil)`) has fewer axes in front of it than the image.
    c_dim = _direct_image_coil_dim(acq_data)

    # `𝒜` always bakes sensitivity composition in when `smaps` is present
    # (`_compose_with_sensitivity`); rebuild the bare (sensitivity-free) encoding operator so
    # per-coil images stay correctly zero-filled/gridded even for a Cartesian-subsampled
    # acquisition, then dispatch the combination explicitly.
    ℬ = isnothing(smaps) ? 𝒜 : get_encoding_operator(CartesianAcquisitionInfo(acq_data; sensitivity_maps = nothing))
    coil_imgs = unname(ℬ' * _measurement(acq_data.kspace_data))

    img_out, coil_reduced = if method.coil_combination isa RootSumSquares
        sqrt.(sum(abs2, coil_imgs; dims = c_dim)), true
    elseif method.coil_combination isa NoCoilCombination
        coil_imgs, false
    else
        throw(ArgumentError("Unsupported coil combination: $(typeof(method.coil_combination))"))
    end
    return _pf_finalize(acq_data, img_out, coil_reduced, c_dim)
end
"""
    _direct_coil_dim(acq::NonCartesianAcquisitionInfo)

Integer position of the coil axis in the *image* a sensitivity-free gridding adjoint produces, or
`0` when the acquisition is single-channel. The non-Cartesian adjoint maps samples to the image
grid, so the coil axis sits directly after the spatial dimensions rather than wherever it lives in
the (sample-indexed) k-space array.
"""
function _direct_coil_dim(acq::NonCartesianAcquisitionInfo)
    isnothing(acq.kspace_data) && return 0
    if _has_dimnames(acq.kspace_data)
        :coil ∈ dimnames(acq.kspace_data) || return 0
    else
        # A per-frame trajectory's frame axes end the k-space, so only a k-space with more axes than
        # samples and frames together has a coil axis.
        nframe = _trajectory_frame_dims_count(acq.trajectory, acq.kspace_data)
        ndims(acq.kspace_data) > _get_sample_dims_count(acq) + nframe || return 0
    end
    return length(acq.image_size) + 1
end

function _direct_reconstruct_coil_combined(acq_data::NonCartesianAcquisitionInfo, method::DirectReconstruction, 𝒜)
    smaps = acq_data.sensitivity_maps
    c_dim = _direct_coil_dim(acq_data)
    if c_dim == 0 || (method.coil_combination isa AdjointSensitivity)
        # Single-channel data, or the combination `𝒜'` already performs: nothing extra to do.
        return 𝒜' * _measurement(acq_data.kspace_data)
    end
    # As in the Cartesian case: `𝒜` composes the sensitivity operator in whenever maps are
    # present, so the per-coil gridded images need a freshly built, sensitivity-free operator.
    ℬ = isnothing(smaps) ? 𝒜 : get_encoding_operator(AcquisitionInfo(acq_data; sensitivity_maps = nothing))
    coil_imgs = unname(ℬ' * _measurement(acq_data.kspace_data))
    img_out, coil_reduced = if method.coil_combination isa RootSumSquares
        sqrt.(sum(abs2, coil_imgs; dims = c_dim)), true
    elseif method.coil_combination isa NoCoilCombination
        coil_imgs, false
    else
        throw(ArgumentError("Unsupported coil combination: $(typeof(method.coil_combination))"))
    end
    return _pf_finalize(acq_data, img_out, coil_reduced, c_dim)
end

function _direct_reconstruct(𝒜, acq_data, x₀, method::ReconstructionMethod, config; scale_override = nothing)
    direct_recon_only = method isa DirectMethod
    if !isnothing(x₀) && direct_recon_only
        log_message(
            config.verbosity,
            "Warning: Initial guess x₀ is ignored when no regularization is specified.",
        )
        x₀ = nothing
    end
    is_default_iterative_adjoint = false
    if isnothing(x₀)
        @step (direct_recon_only ? "Reconstructing image" : "Getting initial estimate") config begin
            if method isa DirectReconstruction
                x₀ = _direct_reconstruct_coil_combined(acq_data, method, 𝒜)
            elseif !(method isa DirectMethod)
                x₀ = 𝒜' * _measurement(acq_data.kspace_data)
                is_default_iterative_adjoint = true
            else
                x₀ = _direct_reconstruct(
                    acq_data, method; progress = progress_tick(config.verbosity)
                )
            end
        end
    end
    scale = _resolve_scale(𝒜, x₀, acq_data, method, config, scale_override)
    # Only the case that actually produced a fresh default adjoint here is rescaled: not a
    # caller-supplied x₀, and not a pure direct method's own reconstruction, which is already
    # correctly scaled.
    is_default_iterative_adjoint || return x₀, scale, _NO_PRIOR
    x̂, L, curvature = _scale_default_warm_start(𝒜, x₀, _adjoint_measurement(acq_data), method, config)
    return x̂, scale, _warm_start_prior(L, curvature)
end

struct NFFTOp{
        T,
        D,
        N,
        M,
        P,
        K <: AbstractArray{Complex{T}},
        DC <: AbstractArray{T},
        IB <: Union{Nothing, AbstractArray{Complex{T}}},
    } <: LinearOperator
    plan::P
    dim_in::NTuple{N, Int}
    dim_out::NTuple{M, Int}
    dims::UnitRange{Int}
    nframe::Int
    ksp_buffer::K
    img_buffer::IB
    dcf::DC
    threaded::Bool
    normal_op::Base.RefValue{Any}
end

"""
	NFFTOp(dim_in::NTuple{N,Int}, trajectory::AbstractArray{T}, dcf::Union{Nothing,Symbol,AbstractArray}=nothing;
	       dims = 1:N, nframe::Int = 0, threaded::Bool = true, array_type::Type = Array{T}, kwargs...)

Create a non-uniform fast Fourier transform operator [1]. The operator is created with a given input
size, trajectory, and density compensation function (dcf). The dcf, when applied, corrects for the
non-uniform sample density of the trajectory in the *adjoint* direction (`op' * ksp`); the forward
direction (`op * image`) never uses it. The operator can be used to transform images to k-space and
back.

The transform runs over the axes `dims` of the input, a range of consecutive axes, and the samples
take their place in the output: an input of size `(Nx, Ny, C)` with `dims = 1:2` maps to
`(samples..., C)`. Every other axis is a batch axis, and each of its images is transformed alike.
The last `nframe` axes of the input, which must come after `dims`, are frame axes with a
trajectory of their own: `trajectory` is then `(D, samples..., frames...)`, and frame `t` of the
input is transformed with `trajectory[:, .., t]`.

On device storage the whole stack is one transform, each step of it a single kernel launch, and
the Toeplitz normal operator is built once for all frames. On the host the images of the stack are
transformed one after another, each threaded as `threaded` allows; to spread a host stack over
threads instead, apply an operator of one image with `BatchOp`.

<em>To use the operator, the NFFT package must be explicitly imported!</em>

# Arguments
- `dim_in::NTuple{N,Int}`: The size of the input, the image axes `dims` and any batch axes.
- `trajectory::AbstractArray{T}`: The trajectory of the samples in k-space. The first dimension
  of the trajectory must match the number of axes in `dims`. The trajectory must have at least
  two dimensions; its last `nframe` axes are frames and must match the last `nframe` of `dim_in`.
- `dcf::Union{Nothing,Symbol,AbstractArray}=nothing`: Controls density compensation:
  - `nothing` (the default): **no** density compensation is applied — the dcf is an array of ones,
    so `op'` is the *true* mathematical adjoint of `op`. This is what any algorithm that assumes
    `A'` is the adjoint (operator-norm estimation via power iteration, CG/CGNR, ...) requires.
  - `:auto`: estimate the dcf of each frame with the iterative sample density compensation method
    of Pipe & Menon [2] (`NFFTTools.sdc`). With `:auto`, `op'` is **not** the true adjoint of `op` —
    it is a density-compensated approximate inverse, useful for a quick direct (gridding)
    reconstruction but wrong as the adjoint fed to an algorithm that relies on the adjoint
    relationship.
  - An `AbstractArray`: used as given (its shape must be that of `trajectory` from the second
    dimension on, and its element type must match `trajectory`'s). Same caveat as `:auto`: a
    non-trivial dcf makes `op'` a weighted approximate inverse, not the true adjoint.
- `dims = 1:N`: The axes of the input the transform runs over, consecutive and ascending.
- `nframe::Int = 0`: The number of trailing input axes that are frame axes of the trajectory.
- `threaded::Bool=true`: `false` disables threading outright; `true` (the default) enables it subject to the threading policy, which also requires more than one Julia thread and CPU storage. Fixed at construction, since the NFFT plan is built for a thread count.
- `array_type::Type = Array{T}`: The storage of the operator's buffers and plans.
- `dcf_estimation_iterations::Int=20`: The number of iterations to use when estimating the dcf.
  Only used when `dcf = :auto`.
- `dcf_correction_function::Function=identity`: A correction function to apply to the estimated dcf.
  Defaults to the identity function. Only used when `dcf = :auto`.
- `kwargs...`: Additional keyword arguments to pass to the NFFTPlan constructor.

# References
1. Fessler, J. A., & Sutton, B. P. (2003). Nonuniform fast Fourier transforms using min-max interpolation.
IEEE Transactions on Signal Processing, 51(2), 560-574.
2. Pipe, J. G., & Menon, P. (1999). Sampling density compensation in MRI: rationale and an iterative numerical solution.

# Examples
```jldoctest
julia> using NFFTOperators

julia> image_size = (128, 128);

julia> trajectory = rand(2, 128, 50) .- 0.5;

julia> dcf = rand(128, 50);

julia> op = NFFTOp(image_size, trajectory, dcf)
𝒩  ℂ^(128, 128) -> ℂ^(128, 50)

julia> image = rand(ComplexF64, image_size);

julia> ksp = op * image;

julia> image_reconstructed = op' * ksp;

julia> NFFTOp((128, 128, 8), trajectory; dims = 1:2)
𝒩  ℂ^(128, 128, 8) -> ℂ^(128, 50, 8)

julia> NFFTOp((128, 128, 8, 4), rand(2, 128, 12, 4) .- 0.5; dims = 1:2, nframe = 1)
𝒩  ℂ^(128, 128, 8, 4) -> ℂ^(128, 12, 8, 4)

```
"""
function NFFTOp(
        dim_in::NTuple{N, Int},
        trajectory::AbstractArray{T},
        dcf::Union{Nothing, Symbol, AbstractArray} = nothing;
        dims = 1:N,
        nframe::Int = 0,
        threaded::Bool = true,
        array_type::Type = Array{T},
        dcf_estimation_iterations::Int = 20,
        dcf_correction_function::Function = identity,
        kwargs...,
    ) where {N, T}
    dims_ = _transform_axes(dims, N)
    D = length(dims_)
    check_traj(trajectory, D)
    0 <= nframe <= ndims(trajectory) - 2 ||
        throw(ArgumentError("a trajectory of $(ndims(trajectory)) axes has no room for $nframe frame axes"))
    last(dims_) <= N - nframe ||
        throw(ArgumentError("the frame axes must come after the transformed axes $dims_"))
    sample_size = size(trajectory)[2:(end - nframe)]
    frame_size = size(trajectory)[(end - nframe + 1):end]
    dim_in[(end - nframe + 1):end] == frame_size || throw(
        DimensionMismatch("the frame axes $(dim_in[(end - nframe + 1):end]) of the input do not match the trajectory's $frame_size")
    )
    image_size = dim_in[dims_]
    inner = dim_in[1:(first(dims_) - 1)]
    outer = dim_in[(last(dims_) + 1):end]
    nbatch = prod(inner; init = 1) * prod(outer[1:(end - nframe)]; init = 1)
    nframes = prod(frame_size; init = 1)
    arr_wrapper = _array_wrapper_type(array_type)
    # Resolved before planning: the plan itself is built for this thread count, so the
    # policy has to have had its say by now (a `nothing` reaching `create_plan` would not
    # even dispatch).
    threaded_flag = _nfft_threaded(threaded, arr_wrapper)
    plan = _nfft_plan(arr_wrapper, trajectory, image_size, threaded_flag; batch = nbatch, frames = nframes, kwargs...)
    dcf_host = _resolve_dcf(
        dcf, plan, trajectory, image_size, sample_size, frame_size, T, dcf_estimation_iterations, dcf_correction_function; kwargs...
    )
    # Buffers and the dcf are in the transform's own layout, `(image or samples, inner..., outer...)`.
    batch_axes = (inner..., outer...)
    dcf_shape = (sample_size..., map(_ -> 1, batch_axes[1:(end - nframe)])..., frame_size...)
    adapted_dcf = _nfft_adapt(arr_wrapper, reshape(collect(T, dcf_host), dcf_shape))
    ksp_buffer = _nfft_adapt(arr_wrapper, zeros(complex(T), sample_size..., batch_axes...))
    img_buffer = isempty(inner) ? nothing : similar(ksp_buffer, (image_size..., batch_axes...))
    dim_out = (inner..., sample_size..., outer...)
    return NFFTOp{T, D, N, length(dim_out), typeof(plan), typeof(ksp_buffer), typeof(adapted_dcf), typeof(img_buffer)}(
        plan, dim_in, dim_out, dims_, nframe, ksp_buffer, img_buffer, adapted_dcf, threaded_flag, Ref{Any}(nothing)
    )
end

function _transform_axes(dims, N)
    d = collect(Int, dims)
    (!isempty(d) && d == first(d):last(d) && 1 <= first(d) && last(d) <= N) ||
        throw(ArgumentError("dims must be consecutive ascending axes of the $N-axis input, got $dims"))
    return first(d):last(d)
end

"""
    _resolve_dcf(dcf, plan, trajectory, image_size, sample_size, frame_size, T, iters, correction; kwargs...)

Resolve the `dcf` keyword of [`NFFTOp`](@ref) into a concrete host dcf array of size
`(sample_size..., frame_size...)`:
- `nothing` -> an array of ones (no density compensation, `op'` is the true adjoint).
- `:auto` -> estimate each frame's with `NFFTTools.sdc`.
- an `AbstractArray` -> used as given, after validating its shape/eltype against `trajectory`.
"""
function _resolve_dcf(::Nothing, plan, trajectory, image_size, sample_size, frame_size, T, iters, correction; kwargs...)
    return ones(T, sample_size..., frame_size...)
end
function _resolve_dcf(dcf::Symbol, plan, trajectory, image_size, sample_size, frame_size, T, iters, correction; kwargs...)
    dcf === :auto || throw(ArgumentError("dcf as a Symbol must be :auto, got :$dcf"))
    if plan isa NFFT.NFFTPlan
        return correction(reshape(NFFTTools.sdc(plan; iters), sample_size))
    end
    D = size(trajectory, 1)
    k = reshape(trajectory, D, :)
    J = prod(sample_size)
    frames = map(1:prod(frame_size; init = 1)) do t
        p = create_plan(k[:, ((t - 1) * J + 1):(t * J)], image_size, false; kwargs...)
        correction(reshape(NFFTTools.sdc(p; iters), sample_size))
    end
    return reshape(stack(frames), sample_size..., frame_size...)
end
function _resolve_dcf(dcf::AbstractArray, plan, trajectory, image_size, sample_size, frame_size, T, iters, correction; kwargs...)
    check_traj_and_dcf(trajectory, dcf, size(trajectory, 1))
    return dcf
end

"""
Minimum Julia thread count before NFFT's threaded path is worth entering.

Unlike every other operator in this package, NFFT's gate is on the *thread count* rather
than on the workload size, because that is what the measurement says decides it.

PROVENANCE: measured. AMD EPYC 7352 (shared), Julia 1.12.7, ComplexF64, 2D trajectory with
`nsamp = s`, `nprof = s ÷ 2`, one process per thread count. Ratios are serial/threaded, so
above 1 means threading wins:

| threads | 48^2 fwd / normal | 96^2 fwd / normal | 192^2 fwd / normal |
|---|---|---|---|
| 2 | 0.59 / 0.85 | 0.78 / 0.99 | 0.91 / 0.90 |
| 3 | 0.87 / 0.81 | 1.03 / 1.07 | 2.29 / 1.41 |
| 4 | 0.87 / 0.96 | 1.93 / 1.56 | 1.43 / 2.21 |

At two workers threading is a loss at *every* size measured, up to a 5 ms `mul!` -- growing
the workload does not rescue it, which is why a size threshold is not the lever this needs.
From three workers up the wins appear and grow with size, so three is the gate.

The 48^2 column stays at or below 1.0 at every thread count and is the obvious candidate for
an additional size gate. It is deliberately *not* added: a repeat of the four-thread run put
48^2 at 1.20 / 1.18 instead, so on this shared machine that column is inside the noise and a
threshold fitted to it would not be measurement, only curve-fitting. The two-thread row, by
contrast, reproduces. Adjoint ratios are omitted from the table for the same reason -- they
ranged from 0.57 to 1.10 across repeats of the same configuration.

The threaded path also allocates 4-12 KiB per `mul!` (FFTW's Julia threading backend spawns
tasks per execution) against 112 B for the serial one, so below the gate it was paying that
for a slowdown. Leaving the gate at `nthreads() > 1` cost 0.63x on the two-thread
`normaloperators/NFFTOp/mul` benchmark with an 18x allocation increase.
"""
const MIN_THREADS_FOR_NFFT = 3

"""
	_nfft_threaded(threaded, arr_wrapper) -> Bool

Resolve NFFT's `threaded` keyword under the package-wide rule: `false` vetoes, `true`
enables subject to policy (see `AbstractOperators._resolve_threaded`).

The policy here is the CPU check plus [`MIN_THREADS_FOR_NFFT`](@ref). There is deliberately
**no size gate**: the transform is planned for a specific trajectory rather than a plain
array length, so there is no single element count to threshold on -- and the sweep behind
`MIN_THREADS_FOR_NFFT` shows size is not what decides it anyway.
"""
function _nfft_threaded(threaded::Bool, arr_wrapper)
    return _resolve_threaded(threaded) do
        Threads.nthreads() >= MIN_THREADS_FOR_NFFT && arr_wrapper === Array
    end
end


# Default (CPU) implementations — overridden by NFFTOperatorsGPUArraysExt for GPU types. A host
# plan transforms one image: frames with trajectories of their own get a plan each, and a batch
# is a loop over the images (`_transform!`).
function _nfft_plan(::Type{Array}, trajectory, image_size, threaded; batch = 1, frames = 1, kwargs...)
    frames == 1 && return create_plan(trajectory, image_size, threaded; kwargs...)
    k = reshape(trajectory, size(trajectory, 1), :)
    J = size(k, 2) ÷ frames
    return [create_plan(k[:, ((t - 1) * J + 1):(t * J)], image_size, threaded; kwargs...) for t in 1:frames]
end
_nfft_adapt(::Type{Array}, arr::AbstractArray) = collect(arr)

"""
    with_nfft_threading(f, threaded::Bool)

Run `f()` with NFFT's, BLAS's and FFTW's threading turned on or off together.

`threaded = true` actively *enables* them rather than merely leaving them alone, because
`NFFT._use_threads[]` defaults to off and is what the NFFT plan consults. Both directions
go through `NestedThreading`, so the request is clamped by any outer budget: an operator
constructed with `threaded = true` that ends up being called from inside a saturated batch
loop stays single-threaded, and the save/restore bookkeeping is refcounted rather than
per-call.
"""
with_nfft_threading(f::F, threaded::Bool) where {F} =
    threaded ? with_full_threads(f) : with_restricted_threads(f)


# The number of images in the stack, and of frames among them, and the size of one image. A
# tuple indexed by a range has no inferable length, so the sizes are gathered element by element.
_nframes(op::NFFTOp{T, D, N}) where {T, D, N} = prod(i -> op.dim_in[i], (N - op.nframe + 1):N; init = 1)
_nimages(op::NFFTOp) = prod(op.dim_in; init = 1) ÷ prod(i -> op.dim_in[i], op.dims; init = 1)
_image_size(op::NFFTOp{T, D}) where {T, D} = ntuple(i -> op.dim_in[first(op.dims) + i - 1], Val(D))

# The axis order of the transform's own layout, `(image, inner..., outer...)`, in the input, and
# that of the output in the transform's `(samples, inner..., outer...)`.
_in_perm(op::NFFTOp{T, D, N}) where {T, D, N} =
    (op.dims..., 1:(first(op.dims) - 1)..., (last(op.dims) + 1):N...)
function _out_perm(op::NFFTOp{T, D, N, M}) where {T, D, N, M}
    pre = first(op.dims) - 1
    ns = M - N + D
    return ((ns + 1):(ns + pre)..., 1:ns..., (ns + pre + 1):M...)
end

function mul!(ksp::AbstractArray, op::NFFTOp, img::AbstractArray)
    AbstractOperators.check(ksp, op, img)
    if op.img_buffer === nothing
        _transform!(ksp, op, img)
    else
        permutedims!(op.img_buffer, img, _in_perm(op))
        _transform!(op.ksp_buffer, op, op.img_buffer)
        permutedims!(ksp, op.ksp_buffer, _out_perm(op))
    end
    return ksp
end

function mul!(
        img::AbstractArray,
        adjop::AdjointOperator{<:NFFTOp},
        ksp::AbstractArray,
    )
    AbstractOperators.check(img, adjop, ksp)
    op = adjop.A
    if op.img_buffer === nothing
        _weight!(op, ksp)
        _adjoint_transform!(img, op, op.ksp_buffer)
    else
        permutedims!(op.ksp_buffer, ksp, invperm(_out_perm(op)))
        _weight!(op, op.ksp_buffer)
        _adjoint_transform!(op.img_buffer, op, op.ksp_buffer)
        permutedims!(img, op.img_buffer, invperm(_in_perm(op)))
    end
    return img
end

function _weight!(op::NFFTOp{T, D, N, M, P, K}, ksp) where {T, D, N, M, P, K}
    # FastBroadcast takes equal axes only; a stack's dcf repeats along its batch axes.
    if !(K <: Array) || size(op.dcf) != size(ksp)
        op.ksp_buffer .= ksp .* op.dcf
    elseif op.threaded
        @.. thread = true op.ksp_buffer = ksp * op.dcf
    else
        @.. op.ksp_buffer = ksp * op.dcf
    end
    return op.ksp_buffer
end

# The transform of a stack in the transform's own layout. One plan does the whole stack on a
# device; on the host each image is transformed by its frame's plan in turn.
function _transform!(ksp, op::NFFTOp, img)
    with_nfft_threading(op.threaded) do
        _each_image(op, ksp, img) do p, k, x
            mul!(k, p, x)
        end
    end
    return ksp
end

function _adjoint_transform!(img, op::NFFTOp, ksp)
    with_nfft_threading(op.threaded) do
        _each_image(op, ksp, img) do p, k, x
            mul!(x, p', k)
        end
    end
    return img
end

function _each_image(f::F, op::NFFTOp{T, D, N, M, P, K}, ksp, img) where {F, T, D, N, M, P, K}
    if !(K <: Array)
        f(op.plan, ksp, img)
        return nothing
    end
    image_size = _image_size(op)
    nf = _nframes(op)
    nb = _nimages(op) ÷ nf
    if nb * nf == 1
        f(op.plan, vec(ksp), reshape(img, image_size))
        return nothing
    end
    ks = reshape(ksp, :, nb, nf)
    xs = reshape(img, :, nb, nf)
    for t in 1:nf
        p = op.plan isa AbstractVector ? op.plan[t] : op.plan
        for b in 1:nb
            f(p, view(ks, :, b, t), reshape(view(xs, :, b, t), image_size))
        end
    end
    return nothing
end

# Properties

size(L::NFFTOp) = L.dim_out, L.dim_in
fun_name(::NFFTOp) = "𝒩"
domain_type(::NFFTOp{T}) where {T} = complex(T)
codomain_type(::NFFTOp{T}) where {T} = complex(T)
domain_array_type(op::NFFTOp{T, D, N, M, P, K}) where {T, D, N, M, P, K} = _array_wrapper_type(K){domain_type(op)}
codomain_array_type(op::NFFTOp{T, D, N, M, P, K}) where {T, D, N, M, P, K} = _array_wrapper_type(K){codomain_type(op)}

# Utility

function check_traj(traj, D)
    @assert size(traj, 1) == D "The first dimension of the trajectory must match the number of image dimensions"
    return @assert ndims(traj) > 1 "The trajectory must have at least two dimensions"
end

function check_traj_and_dcf(traj, dcf, D)
    check_traj(traj, D)
    @assert tuple(size(traj)[2:end]...) == size(dcf) "Shape of the trajectory from the second dimension must match the shape of the dcf array"
    return @assert eltype(traj) == eltype(dcf) "The element type of the trajectory must match the element type of the dcf array"
end

function create_plan(trajectory, image_size, threaded; kwargs...)
    traj = reshape(trajectory, size(trajectory, 1), :)
    return with_nfft_threading(threaded) do
        NFFTPlan(traj, image_size; kwargs...)
    end
end

# Helper to create matched forward/backward FFT plans so that JET can track
# that both plans have the same element type T and dimension D.
struct _MatchedFFTPlans{T, D}
    forward::FFTW.cFFTWPlan{Complex{T}, -1, true, D, UnitRange{Int64}}
    backward::FFTW.cFFTWPlan{Complex{T}, 1, true, D, UnitRange{Int64}}
end

function _make_matched_fft_plans(tmpVec::Array{Complex{T}, D}, dims_; kwargs...) where {T, D}
    FP = FFTW.plan_fft!(tmpVec, dims_; kwargs...)::FFTW.cFFTWPlan{Complex{T}, -1, true, D, UnitRange{Int64}}
    BP = FFTW.plan_bfft!(tmpVec, dims_; kwargs...)::FFTW.cFFTWPlan{Complex{T}, 1, true, D, UnitRange{Int64}}
    return _MatchedFFTPlans{T, D}(FP, BP)
end

function NFFTPlan(
        k::Matrix{T},
        N::NTuple{D, Int};
        dims::Union{Integer, UnitRange{Int64}} = 1:D,
        fftflags = nothing,
        kwargs...,
    ) where {T, D}
    NFFT.checkNodes(k)

    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)

    if length(NOut) > 1
        params.precompute = NFFT.LINEAR
    end

    tmpVec = Array{Complex{T}, D}(undef, Ñ)

    fftflags_ = (fftflags !== nothing) ? (flags = fftflags,) : NamedTuple()
    plans = _make_matched_fft_plans(tmpVec, dims_; num_threads = FFTW.get_num_threads(), fftflags_...)
    FP = plans.forward
    BP = plans.backward

    calcBlocks =
        (
        params.precompute == NFFT.LINEAR ||
            params.precompute == NFFT.TENSOR ||
            params.precompute == NFFT.POLYNOMIAL
    ) &&
        params.blocking &&
        length(dims_) == D

    blocks, nodesInBlocks, blockOffsets, idxInBlock, windowTensor = NFFT.precomputeBlocks(
        k, Ñ, params, calcBlocks
    )

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(
        k, N[dims_], Ñ[dims_], params
    )

    U = params.storeDeconvolutionIdx ? N : ntuple(d -> 0, D)
    tmpVecHat = Array{Complex{T}, D}(undef, U)

    return NFFT.NFFTPlan(
        N,
        NOut,
        J,
        k,
        Ñ,
        dims_,
        params,
        FP,
        BP,
        tmpVec,
        tmpVecHat,
        deconvolveIdx,
        windowHatInvLUT,
        windowLinInterp,
        windowPolyInterp,
        blocks,
        nodesInBlocks,
        blockOffsets,
        idxInBlock,
        windowTensor,
        B,
    )
end


# ─── Threading ────────────────────────────────────────────────────────────────
#
# NFFT is a *counted* pool in NestedThreading: the plan itself is built for a thread count
# and `mul!` runs under `with_full_threads`/`with_restricted_threads` (see `_nfft_run`).
# So `threaded` here selects a plan-time thread count, not a Julia loop, and it cannot be
# flipped on an existing plan -- but it does not need a full replan either: `copy_operator`
# copies the plan under the target budget, which rebuilds only its FFT plans (see
# `_copy_operator_impl`).
is_threaded(op::NFFTOp) = op.threaded
supports_threading(::NFFTOp) = true

# Not thread-safe: `ksp_buffer` is operator-owned scratch written by every `mul!`.
is_thread_safe(::NFFTOp) = false

# A host plan's FFT plans and grid are its own scratch; a copy gets its own, with the gridding
# tables copied rather than recomputed. A device plan runs in its stream's order and is shared.
_copy_plan(p::NFFT.NFFTPlan) = copy(p)
_copy_plan(ps::AbstractVector) = map(_copy_plan, ps)
_copy_plan(p) = p

# The nodes of all frames, `(D, samples × frames)`.
_nodes(op::NFFTOp) = op.plan isa AbstractVector ? reduce(hcat, [p.k for p in op.plan]) : op.plan.k
_first_plan(op::NFFTOp) = op.plan isa AbstractVector ? first(op.plan) : op.plan

function _copy_operator_impl(
        op::NFFTOp{T, D, N, M, P, K, DC, IB}; storage_type = nothing, threaded = nothing
    ) where {T, D, N, M, P, K, DC, IB}
    if storage_type === nothing
        # `mul!` writes the plan's internal scratch (`plan.tmpVec` / `plan.tmpVecHat`), so a
        # plan *shared* between two operator copies races when they run concurrently -- which
        # is exactly what a per-thread copy is for. The copy is made inside the target threading
        # scope, since it re-plans the FFTs at FFTW's *current* budget, and nothing else in the
        # plan is thread-count-specific. `threaded` still goes through the package rule (`false`
        # vetoes, `true` is a permission), so `is_threaded` on the copy reports what it will
        # actually do.
        #
        # `dcf` is read-only in `mul!` and is shared per the copy convention; the buffers are
        # operator-owned scratch, so the copy gets its own.
        new_threaded = if threaded === nothing
            op.threaded
        else
            _nfft_threaded(threaded, _array_wrapper_type(K))
        end
        plan = with_nfft_threading(new_threaded) do
            _copy_plan(op.plan)
        end
        img_buffer = op.img_buffer === nothing ? nothing : similar(op.img_buffer)
        return NFFTOp{T, D, N, M, P, K, DC, IB}(
            plan, op.dim_in, op.dim_out, op.dims, op.nframe, similar(op.ksp_buffer), img_buffer, op.dcf,
            new_threaded, Ref{Any}(nothing),
        )
    end
    new_threaded = threaded === nothing ? op.threaded : threaded
    # Storage-backend change: rebuild on the requested array type from the trajectory, which
    # the plans still carry (`plan.k`, in the flattened 2D form `create_plan` reshapes to).
    frame_size = op.dim_in[(end - op.nframe + 1):end]
    sample_size = op.dim_out[first(op.dims):(first(op.dims) + M - N + D - 1)]
    trajectory = reshape(collect(_nodes(op)), D, sample_size..., frame_size...)
    dcf = reshape(collect(op.dcf), sample_size..., frame_size...)
    # The gridding operating point is not recoverable from the trajectory, so it has to be read
    # off the old plan and forwarded: rebuilding at the constructor defaults would silently give
    # the copy a *different* transform from the one the caller set up.
    return NFFTOp(
        op.dim_in, trajectory, dcf;
        dims = op.dims, nframe = op.nframe, threaded = new_threaded, array_type = storage_type{T},
        _operating_point_kwargs(_first_plan(op))...,
    )
end

"""
    _operating_point_kwargs(plan) -> NamedTuple

The gridding operating point (`m`, `σ`, `precompute`) of an existing plan, in the keyword form
[`NFFTOp`](@ref) forwards to `NFFT.initParams`. A plan that does not carry an `NFFTParams` --
some GPU backends wrap their own -- yields an empty tuple, leaving the constructor's defaults in
place, which is the pre-existing behaviour.
"""
function _operating_point_kwargs(plan)
    hasproperty(plan, :params) || return NamedTuple()
    params = plan.params
    return (m = params.m, σ = params.σ, precompute = params.precompute)
end

# The _mul! and _get_normal_op function are based on the implementation of the
# NFFTToeplitzNormalOp from LinearOperatorCollection.jl. That's the reason why
# the license is included here. The original code can be found here:
# https://github.com/JuliaImageRecon/LinearOperatorCollection.jl/blob/main/ext/LinearOperatorNFFTExt/NFFTOp.jl

# MIT License
#
# Copyright (c) 2023 Tobias Knopp <tobias.knopp@tuhh.de> and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

struct NfftNormalOp{N, T, A <: AbstractArray, L <: AbstractArray, F, I, P, X} <: LinearOperator
    array_type::Type{T}
    shape::NTuple{N, Int}
    λ::L
    fftplan::F
    ifftplan::I
    buf::A
    perm::P
    xbuf::X
    threaded::Bool
end

function mul!(y::AbstractArray, op::NfftNormalOp, x::AbstractArray)
    AbstractOperators.check(y, op, x)
    return with_nfft_threading(op.threaded) do
        if op.perm === nothing
            _mul!(y, op, x)
        else
            permutedims!(op.xbuf, x, op.perm)
            _mul!(op.xbuf, op, op.xbuf)
            permutedims!(y, op.xbuf, invperm(op.perm))
        end
    end
end

# `x` and `y` are in the transform's own layout, `(image..., stack...)`; `buf` is the stack on the
# twice-oversampled grid, and `λ` the kernel of each frame, broadcast over the rest of the stack.
# The inverse FFT is unnormalized, its `1/length` folded into `λ`.
function _mul!(y, op::NfftNormalOp, x)
    op.buf .= 0
    op.buf[CartesianIndices(x)] .= x
    op.fftplan * op.buf # in-place FFT
    op.buf .*= op.λ
    op.ifftplan * op.buf # in-place unnormalized IFFT
    y .= @view op.buf[CartesianIndices(x)]
    return y
end

"""
    _planner_rigor(plan) -> flags

The planner rigor (`ESTIMATE`, `MEASURE`, `PATIENT`, `EXHAUSTIVE`, `WISDOM_ONLY`) of the FFT
inside NFFT plan `plan`, so the Toeplitz embedding's FFT is planned as carefully as the
operator's own. The two FFTs run equally often, and `ESTIMATE` can pick a plan several times
slower than `MEASURE` (7x for a 256×256 in-place `ComplexF32` FFT on an EPYC 7352). A plan that
records no FFTW flags (a GPU plan) gets `ESTIMATE`.
"""
function _planner_rigor(plan)
    fft = hasproperty(plan, :forwardFFT) ? plan.forwardFFT : nothing
    fft isa FFTW.FFTWPlan || return FFTW.ESTIMATE
    return fft.flags & (FFTW.ESTIMATE | FFTW.PATIENT | FFTW.EXHAUSTIVE | FFTW.WISDOM_ONLY)
end

# The planner keywords of the embedding's FFTs: FFTW's planner rigor on the host, none for a
# device FFT, which takes no flags.
_embedding_fft_kwargs(op::NFFTOp{T, D, N, M, P, <:Array}) where {T, D, N, M, P} = (flags = _planner_rigor(_first_plan(op)),)
_embedding_fft_kwargs(::NFFTOp) = NamedTuple()

# The normal operator owns FFT plans and a buffer of the whole stack on the twice-oversampled
# grid; it is built on the first request and returned to every later one.
AbstractOperators.has_optimized_normalop(::NFFTOp) = true
function AbstractOperators.get_normal_op(op::NFFTOp)
    cached = op.normal_op[]
    cached === nothing || return cached
    normal = with_nfft_threading(op.threaded) do
        _get_normal_op(op)
    end
    op.normal_op[] = normal
    return normal
end

function _get_normal_op(op::NFFTOp{T, D, N}) where {T, D, N}
    shape = op.dim_in[op.dims]
    shape_ext = 2 .* shape
    perm = _in_perm(op)
    stack = op.dim_in[collect(perm[(D + 1):end])]
    frame_size = op.dim_in[(end - op.nframe + 1):end]
    λ = _toeplitz_kernel(op, shape_ext, _nframes(op))
    λ = reshape(λ, shape_ext..., map(_ -> 1, stack[1:(end - op.nframe)])..., frame_size...)
    buf = similar(op.ksp_buffer, (shape_ext..., stack...))
    kw = _embedding_fft_kwargs(op)
    fftplan = FFTW.plan_fft!(buf, 1:D; kw...)
    ifftplan = FFTW.plan_bfft!(buf, 1:D; kw...)
    fill!(buf, 0)
    xbuf = op.img_buffer === nothing ? nothing : similar(op.img_buffer)
    return NfftNormalOp(
        domain_array_type(op), op.dim_in, λ, fftplan, ifftplan, buf,
        op.img_buffer === nothing ? nothing : perm, xbuf, op.threaded,
    )
end

# The Toeplitz kernel of each of `frames` frames, `(2N..., frames)`: the FFT of the adjoint NFFT
# of the frame's density compensation on the twice-oversampled grid, centred, and divided by the
# grid size for the unnormalized inverse.
function _toeplitz_kernel(op::NFFTOp{T, D}, shape_ext, frames) where {T, D}
    k = _nodes(op)
    params = _first_plan(op).params
    J = size(k, 2) ÷ frames
    w = reshape(Complex{T}.(op.dcf), J, frames)
    λ = zeros(Complex{T}, shape_ext..., frames)
    λs = reshape(λ, :, frames)
    for t in 1:frames
        p = NFFTPlan(
            k[:, ((t - 1) * J + 1):(t * J)], shape_ext;
            m = params.m, σ = params.σ, precompute = NFFT.POLYNOMIAL, fftflags = FFTW.ESTIMATE, blocking = true,
        )
        mul!(reshape(view(λs, :, t), shape_ext), adjoint(p), w[:, t])
    end
    λ = circshift(λ, (shape_ext .÷ 2..., 0))
    FFTW.fft!(λ, 1:D)
    λ ./= T(prod(shape_ext))
    return λ
end

# properties

Base.size(op::NfftNormalOp) = op.shape, op.shape
AbstractOperators.fun_name(::NfftNormalOp) = "(𝒩ᵃ𝒩)"
domain_type(::NfftNormalOp{N, T}) where {N, T} = eltype(T)
codomain_type(::NfftNormalOp{N, T}) where {N, T} = eltype(T)
domain_array_type(op::NfftNormalOp{N, T}) where {N, T} = op.array_type
codomain_array_type(op::NfftNormalOp{N, T}) where {N, T} = op.array_type
is_symmetric(::NfftNormalOp) = true
AdjointOperator(op::NfftNormalOp) = op

"""
    L1Shearlet(λ; directions=(8, 8, 16))

Create an ℓ₁ shearlet regularization term for images with parameter `λ`: `λ‖𝒮x‖₁`, where `𝒮`
is the band-limited discrete shearlet transform `FFTWOperators.ShearletOp`
applied to the leading 2D image plane, and to every 2D slice independently when the image has
more dimensions (time, slices, ...), as [`L1Wavelet2D`](@ref) does.

Shearlets are a directional multiscale frame with parabolic scaling: each of the
`length(directions)` dyadic scales (coarse to fine) is split into `directions[j]` orientations,
so edges and curves are represented by fewer large coefficients than with separable wavelets.
The transform is a Parseval frame (`𝒮ᴴ𝒮 = I`) but redundant (`1 + sum(directions)` bands, all
the size of the image), so the term has no closed-form proximal step and the reconstruction
uses a splitting method (ADMM) for it.

# Arguments
- `λ`: Regularization parameter (scalar).
- `directions`: (optional) Number of orientations per scale, coarse to fine. Default `(8, 8, 16)`.
"""
struct L1Shearlet{T <: Real, D <: Tuple} <: Regularization
    λ::T
    directions::D
    function L1Shearlet(λ::Real; directions::Tuple = (8, 8, 16))
        @argcheck all(>(0), directions) "every direction count must be positive"
        return new{typeof(λ), typeof(directions)}(λ, directions)
    end
end

"""
    L1Surfacelet(λ; levels=(1, 1, 1))

Create an ℓ₁ surfacelet regularization term for volumes with parameter `λ`: `λ‖𝒮x‖₁`, where
`𝒮` is the undecimated band-limited surfacelet transform
`FFTWOperators.SurfaceletOp` applied to the leading three dimensions, and
to every volume independently when the image has more dimensions, as [`L1Wavelet3D`](@ref) does.

Surfacelets (Lu and Do, 2007) are the 3D counterpart of contourlets: each of the
`length(levels)` dyadic scales is split into three hourglass-shaped pyramids by the dominant
frequency axis, and each pyramid into `2^l × 2^l` directional subbands (`l = levels[j]`), so
surface-like singularities are captured sparsely. This implementation applies the surfacelet
frequency windows by FFT without subsampling: a shift-invariant Parseval frame (`𝒮ᴴ𝒮 = I`) with
`1 + sum(3 .* 4 .^ levels)` bands, all the size of the volume. Like [`L1Shearlet`](@ref) the
term has no closed-form proximal step and is handled by ADMM.

The third dimension may be time (e.g. cardiac phases): the FFT treats it as periodic, which
suits cyclic motion bins.

# Arguments
- `λ`: Regularization parameter (scalar).
- `levels`: (optional) Directional level per scale, coarse to fine. Default `(1, 1, 1)`
  (12 directions per scale).
"""
struct L1Surfacelet{T <: Real, L <: Tuple} <: Regularization
    λ::T
    levels::L
    function L1Surfacelet(λ::Real; levels::Tuple = (1, 1, 1))
        @argcheck all(>=(0), levels) "directional levels must be non-negative"
        return new{typeof(λ), typeof(levels)}(λ, levels)
    end
end

function get_operator(reg::L1Shearlet, x::AbstractArray{T}; threaded::Bool = true) where {T}
    @argcheck ndims(x) >= 2 "L1Shearlet requires at least 2 dimensions in the input variable"
    _is_device(x) && return _host_staged(t -> get_operator(reg, t; threaded), x)
    ximg = x isa NamedDimsArray ? parent(x) : x
    op = ShearletOp(T, size(ximg)[1:2]; directions = reg.directions, threaded)
    return ndims(ximg) > 2 ? BatchOp(op, size(ximg)[3:end]; threaded) : op
end

function get_operator(reg::L1Surfacelet, x::AbstractArray{T}; threaded::Bool = true) where {T}
    @argcheck ndims(x) >= 3 "L1Surfacelet requires at least 3 dimensions in the input variable"
    _is_device(x) && return _host_staged(t -> get_operator(reg, t; threaded), x)
    ximg = x isa NamedDimsArray ? parent(x) : x
    op = SurfaceletOp(T, size(ximg)[1:3]; levels = reg.levels, threaded)
    return ndims(ximg) > 3 ? BatchOp(op, size(ximg)[4:end]; threaded) : op
end

get_affected_dims(::L1Shearlet, ::Nothing, image_dims) = image_dims[1:2]
get_affected_dims(::L1Surfacelet, ::Nothing, image_dims) = image_dims[1:3]

# L1 norm is homogeneous of degree 1, so λ scales linearly (see scale_regularization docstring).
scale_regularization(reg::L1Shearlet, factor::Real) = L1Shearlet(reg.λ * factor; directions = reg.directions)
scale_regularization(reg::L1Surfacelet, factor::Real) = L1Surfacelet(reg.λ * factor; levels = reg.levels)

function materialize(reg::Union{L1Shearlet, L1Surfacelet}, x::Variable{T}; threaded::Bool) where {T}
    𝒮 = get_operator(reg, ~x; threaded)
    λ = real(T)(reg.λ)
    repr = @sprintf "%g ⋅ ‖𝒮%s‖₁" λ get_name(x)
    return StructuredOptimization.Term(1, NormL1(λ; threaded), 𝒮 * x, repr)
end

# MRI Operators

The operators that model the encoding: Fourier transform, coil sensitivity maps and k-space
subsampling, and their composition. [The low-level interface](../tutorials/12_low_level_interface.md)
builds them, checks their adjoints and uses them in a hand-written problem.

!!! note "These names are public, but not exported"
    The operator constructors below belong to the extension surface rather than to the beginner-facing
    API, so `using Ristretto` does not bring them into scope. Import the ones you need
    explicitly:

    ```julia
    using Ristretto: get_encoding_operator, get_fourier_operator,
        get_sensitivity_map_operator, get_subsampling_operator
    ```

## Overview

The MRI encoding operator $\mathcal{A}$ maps an image to k-space measurements:

```math
y = \mathcal{A} x = \mathcal{P} \mathcal{F} \mathcal{S} x
```

where:
- `𝒮`: Sensitivity map operator (coil sensitivities)
- `ℱ`: Fourier transform operator
- `𝒫`: Subsampling operator (k-space sampling pattern)
- `x`: Image (single-coil)
- `y`: Acquired k-space data (multi-coil, potentially undersampled)

## API Reference

```@docs
get_encoding_operator
get_fourier_operator
get_sensitivity_map_operator
Ristretto.get_subsampling_operator
```

## Encoding Operators

```@setup operators
using Ristretto
using Ristretto: get_encoding_operator
n, nc = 64, 8
ky = falses(n); ky[1:2:n] .= true
mask = (:, ky)
sensitivity_maps = coil_sensitivities(n, n, nc)
ksp = rand(ComplexF32, n, count(ky), nc)
acq = AcquisitionInfo(ksp; is3D = false, sensitivity_maps, image_size = (n, n), subsampling = mask)
x = rand(ComplexF32, n, n)
```

`get_encoding_operator` builds ``\mathcal{A}`` from an `AcquisitionInfo`, or from k-space data
with the same keywords `AcquisitionInfo` takes:

```@example operators
𝒜 = get_encoding_operator(acq)
𝒜 = get_encoding_operator(ksp, false; sensitivity_maps, image_size = (64, 64), subsampling = mask)
x̂ = 𝒜' * (𝒜 * x)
nothing # hide
```

Keywords: `sensitivity_maps`, `image_size` (required with a subsampling pattern), `subsampling`,
`threaded = true`, and `fast_planning = false`: the FFTs are planned with `FFTW.MEASURE`, or with
the quicker `FFTW.ESTIMATE` when `fast_planning = true`. With named k-space, the operator maps the dimension names: its domain is named
`:x`, `:y`, `:z`, its codomain `:kx`, `:ky`, `:kz`, and batch names pass through.

## Fourier Operators

`get_fourier_operator(x, is3D)` transforms the first two (or three) dimensions and leaves the
others, coils and batch dimensions, untouched:

```math
k(u,v) = \sum_{x=0}^{N_x-1} \sum_{y=0}^{N_y-1} I(x,y) e^{-2\pi i (ux/N_x + vy/N_y)}
```

Its adjoint is the inverse transform, with the factor ``1/(N_x N_y)``, so it is the true adjoint only
up to that factor.

## Sensitivity Map Operators

`get_sensitivity_map_operator(smaps, is3D; batch_dims)` maps a single image to one image per coil,
and its adjoint combines them:

```math
(S x)_c = s_c \odot x, \qquad S^H y = \sum_{c=1}^{N_c} \bar{s}_c \odot y_c
```

| Acquisition | Map shape |
|---|---|
| 2D | `(nx, ny, ncoil)` |
| 2D multi-slice, per-slice maps | `(nx, ny, ncoil, nslice)` |
| 3D | `(nx, ny, nz, ncoil)` |
| any, with batch dimensions the maps do not have | the maps above plus `batch_dims = (nt, ...)` |

Named maps are `(:x, :y, [:z], :coil)`, in that order.

## Subsampling Operators

`get_subsampling_operator` keeps the sampled k-space locations; its adjoint zero-fills the rest.
The pattern takes any of these forms:

| Pattern | Example |
|---|---|
| boolean mask over the encoded axes | `mask::Array{Bool,2}` or `Array{Bool,3}` |
| linear or Cartesian indices | `Vector{Int}`, `Vector{CartesianIndex{2}}` |
| one pattern per axis | `(1:64, mask_ky)`, `(1:64, mask_ky, 1:2:32)` |
| a mask over two axes and a pattern for the third | `(mask_xy, 1:2:32)`, `(1:64, mask_yz)` |

## Contourlet Operators

`L1Contourlet` is built on the nonsubsampled contourlet transform from `ContourletOperators.jl`.
`StackedNSCTOp` is the wrapper this package adds so that the NSCT bands, which the bare operator
returns as a ragged tuple, come back as one array a `NormL1` term can be applied to.

```@docs
ContourletOperators.NSCTOp
ContourletOperators.ContourletOp
Ristretto.StackedNSCTOp
```

## See Also

- [AcquisitionInfo](../high-level/acquisition_info.md) - High-level configuration
- [Reconstruction](../high-level/reconstruction.md) - Using operators in reconstruction
- [Simulation Tools](../high-level/simulation.md) - Generate test data and patterns

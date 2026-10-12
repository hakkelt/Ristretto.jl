# MRI Operators

*Tutorial: [Low-level interface](../tutorials/12_low_level_interface.md).*

This page documents the low-level operator interface for MRI reconstruction. These operators model the physical MRI encoding process and its components: Fourier transforms, coil sensitivity maps, and k-space subsampling patterns.

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

The encoding operator is the primary interface for creating complete MRI forward models.

```@setup ops
using Ristretto
using Random
Random.seed!(123)
```

### Usage Patterns

```@example ops
using Ristretto
using Ristretto: get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator

# Single-coil, fully sampled
ksp = rand(ComplexF32, 64, 64)
get_encoding_operator(ksp, false)
```

```@example ops
# Multi-coil parallel imaging
ksp_mc = rand(ComplexF32, 64, 64, 8)
sensitivity_maps = coil_sensitivities(64, 64, 8)
get_encoding_operator(ksp_mc, false; sensitivity_maps=sensitivity_maps)
```

```@example ops
# Undersampled reconstruction
mask = rand(Bool, 64, 64)
mask[28:36, 28:36] .= true  # Fully sample center
ksp_sub = rand(ComplexF32, sum(mask), 8)

get_encoding_operator(
    ksp_sub, false; 
    sensitivity_maps=sensitivity_maps,
    image_size=(64, 64), 
    subsampling=mask
)
```

### With Named Dimensions

```@example ops
using NamedDims

ksp_named = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 64, 64, 8))
smaps_named = NamedDimsArray{(:x, :y, :coil)}(coil_sensitivities(64, 64, 8))

get_encoding_operator(ksp_named; sensitivity_maps=smaps_named)
```

### Keyword Arguments

All encoding operator methods support:

- `sensitivity_maps=nothing`: Sensitivity maps for parallel imaging
- `image_size=nothing`: Image size (required for subsampled reconstruction)
- `subsampling=nothing`: Subsampling pattern
- `threaded::Bool=true`: Enable multi-threading
- `fast_planning::Bool=false`: Use fast FFTW planning

## Fourier Operators

Fourier operators implement the discrete Fourier transform between image space and k-space.

### Mathematical Background

**Forward transform (image → k-space):**
```math
k(u,v) = \sum_{x=0}^{N_x-1} \sum_{y=0}^{N_y-1} I(x,y) e^{-2\pi i (ux/N_x + vy/N_y)}
```

**Inverse transform (k-space → image):**
```math
I(x,y) = \frac{1}{N_x N_y} \sum_{u=0}^{N_x-1} \sum_{v=0}^{N_y-1} k(u,v) e^{2\pi i (ux/N_x + vy/N_y)}
```

### Basic Usage

```@example ops
# 2D Fourier transform
img = rand(ComplexF32, 64, 64)
F = get_fourier_operator(img, false)  # false = 2D

ksp_fft = F * img        # Forward FFT
img_recon = F' * ksp_fft # Inverse FFT
nothing # hide
```

```@example ops
# 3D Fourier transform
img_3d = rand(ComplexF32, 64, 64, 32)
F_3d = get_fourier_operator(img_3d, true)  # true = 3D
nothing # hide
```

```@example ops
# Multi-coil data (FFT applied per coil)
ksp_multi = rand(ComplexF32, 64, 64, 8)
F_multi = get_fourier_operator(ksp_multi, false)
img_multi = F_multi' * ksp_multi  # Shape: (64, 64, 8)
nothing # hide
```

### Dimension Name Mappings

When using `NamedDimsArray`, dimension names are automatically mapped:

| Image Space | K-space |
|-------------|---------||
| `:x`        | `:kx`   |
| `:y`        | `:ky`   |
| `:z`        | `:kz`   |

```@example ops
# Create k-space data with proper dimension names
ksp_named = NamedDimsArray{(:kx, :ky)}(rand(ComplexF32, 64, 64))
F_named = get_fourier_operator(ksp_named, false)

# The operator preserves dimension names during inverse FFT
img_named = F_named' * ksp_named
println("K-space dimensions: ", dimnames(ksp_named))
println("Image dimensions: ", dimnames(img_named))
```

### Implementation Details

- **FFTW Integration**: Uses FFTW.jl for high-performance computation
  - `FFTW.MEASURE` (default): Optimal performance with longer planning
  - `FFTW.ESTIMATE`: Faster planning with potentially slower execution
- **Multi-threading**: Automatically uses all available threads when `threaded=true`
- **Dimension Handling**: 
  - 2D: FFT along dimensions (1, 2)
  - 3D: FFT along dimensions (1, 2, 3)
  - Batch dimensions (coils, time) are preserved

## Sensitivity Map Operators

Sensitivity map operators model the spatial sensitivity profiles of receiver coils in parallel MRI.

### Mathematical Formulation

**Forward operation** (single-coil → multi-coil):
```math
(S x)_c = s_c \odot x
```

**Adjoint operation** (multi-coil → single-coil):
```math
S^H y = \sum_{c=1}^{N_c} \bar{s}_c \odot y_c
```

where $s_c$ is the sensitivity map for coil $c$, $\odot$ denotes element-wise multiplication, and $N_c$ is the number of coils.

### Basic Usage

```@example ops
# 2D multi-coil sensitivity maps
nx, ny, nc = 64, 64, 8
smaps = coil_sensitivities(nx, ny, nc)
S = get_sensitivity_map_operator(smaps, false)

# Forward: single-coil → multi-coil
img_single = rand(ComplexF32, nx, ny)
multi_coil = S * img_single
println("Multi-coil size: ", size(multi_coil))  # (64, 64, 8)
```

```@example ops
# Adjoint: multi-coil → combined image
img_combined = S' * multi_coil
println("Combined size: ", size(img_combined))  # (64, 64)
```

### 3D Sensitivity Maps

```@example ops
# 3D sensitivity maps
nx, ny, nz, nc = 64, 64, 32, 8
smaps_3d = coil_sensitivities(nx, ny, nz, nc)
S_3d = get_sensitivity_map_operator(smaps_3d, true)

img_3d = rand(ComplexF32, nx, ny, nz)
multi_coil_3d = S_3d * img_3d
println("3D multi-coil size: ", size(multi_coil_3d))  # (64, 64, 32, 8)
```

### Multi-Slice with Per-Slice Sensitivity Maps

```@example ops
# Per-slice sensitivity maps
nx, ny, nc, nz = 64, 64, 8, 20
smaps_per_slice = repeat(coil_sensitivities(nx, ny, nc), outer=(1, 1, 1, nz))

S_multislice = get_sensitivity_map_operator(smaps_per_slice, false)

img_slices = rand(ComplexF32, nx, ny, nz)
multi_coil_slices = S_multislice * img_slices
println("Multi-slice coil size: ", size(multi_coil_slices))  # (64, 64, 8, 20)
```

### Batch Dimensions (Dynamic Imaging)

```@example ops
# Sensitivity maps for dynamic imaging
nx, ny, nc, nt = 64, 64, 8, 30  # 30 time frames
smaps_dyn = coil_sensitivities(nx, ny, nc)

S_dynamic = get_sensitivity_map_operator(
    smaps_dyn, false; 
    batch_dims=(nt,)
)

img_dynamic = rand(ComplexF32, nx, ny, nt)
multi_coil_dynamic = S_dynamic * img_dynamic
println("Dynamic multi-coil size: ", size(multi_coil_dynamic))  # (64, 64, 8, 30)
```

### Required Array Shapes

**2D Sensitivity Maps:**
- Single slice: `(nx, ny, ncoils)`
- Multi-slice: `(nx, ny, ncoils, nslices)`

**3D Sensitivity Maps:**
- Standard: `(nx, ny, nz, ncoils)`

**NamedDims Requirements:**
- Must include `:x`, `:y` (and `:z` for 3D)
- Must have `:coil` as the last spatial dimension
- Dimension order: `:x`, `:y`, [`:z`], `:coil`

### Physical Interpretation

The adjoint operation $S^H$ performs optimal coil combination:
```math
x_{\text{combined}} = S^H y = \sum_{c=1}^{N_c} \bar{s}_c \odot y_c
```

This weighted combination assigns higher weights to regions where each coil has stronger sensitivity, accounting for spatial response and phase variations.

## Subsampling Operators

Subsampling operators model undersampled k-space acquisition patterns for accelerated MRI.

### Mathematical Formulation

**Forward operation** (extract sampled locations):
```math
y = \Gamma x
```

**Adjoint operation** (zero-fill unsampled locations):
```math
\Gamma^H y = \text{zero-filled full k-space}
```

The adjoint places acquired data at sampled locations and zeros elsewhere.

### Subsampling Pattern Types

#### 2D Patterns

```@example ops
# Boolean mask
mask_bool = rand(Bool, 64, 64)
mask_bool[30:35, 30:35] .= true  # Fully sample center
println("Acceleration: ", round(64*64/sum(mask_bool), digits=2), "x")
```

```@example ops
# Linear indices
indices_linear = [1, 10, 50, 100, 500, 1000]
nothing # hide
```

```@example ops
# Cartesian indices
cart_indices = [CartesianIndex(i, j) for i in 1:64 for j in 1:64 if rand() < 0.3]
nothing # hide
```

```@example ops
# Separable pattern (independent per dimension)
pattern_x = 1:2:64        # Every other line
pattern_y = rand(Bool, 64)  # Random sampling
pattern_sep = (pattern_x, pattern_y)
nothing # hide
```

#### 3D Patterns

```@example ops
# 3D boolean mask
mask_3d = rand(Bool, 64, 64, 32)
mask_3d[30:35, 30:35, 14:18] .= true
nothing # hide
```

```@example ops
# Separable 3D
pattern_3d = (1:64, rand(Bool, 64), 1:2:32)  # (x: all, y: random, z: every other)
nothing # hide
```

```@example ops
# Hybrid 2D+1D (common for 3D Cartesian)
mask_2d = rand(Bool, 64, 64)
pattern_kz = 1:2:32
pattern_hybrid = (mask_2d, pattern_kz)
nothing # hide
```

### Basic Usage

```@example ops
# Create undersampling pattern
nx, ny = 64, 64
mask = rand(Bool, nx, ny)
mask[30:35, 30:35] .= true

# Subsample k-space (use tuple pattern)
ksp_full = rand(ComplexF32, nx, ny, 8)
indices = findall(mask)
ksp_sub = ksp_full[indices, :]

# Create subsampling operator with tuple pattern
𝒫 = get_subsampling_operator(ksp_sub, (nx, ny), (mask,))

# Zero-fill reconstruction
ksp_zerofilled = 𝒫' * ksp_sub
println("Zero-filled size: ", size(ksp_zerofilled))  # (64, 64, 8)
```

### Combined Operations

For complete encoding including Fourier and subsampling, use `get_encoding_operator`:

```@example ops
# Complete encoding operator: Image → Subsampled k-space
img_size = (64, 64)
smaps_enc = coil_sensitivities(64, 64, 8)
E_complete = get_encoding_operator(
    ksp_sub, false;
    sensitivity_maps=smaps_enc,
    image_size=img_size,
    subsampling=(mask,)
)

# Direct operations
img = rand(ComplexF32, 64, 64)
ksp_acquired = E_complete * img        # Forward
img_recon = E_complete' * ksp_acquired # Adjoint
nothing # hide
```

### Multi-Slice 2D Subsampling

```@example ops
# For multi-slice data, the encoding operator handles it automatically
nx, ny, nz = 64, 64, 20
mask_2d = rand(Bool, nx, ny)
indices_2d = findall(mask_2d)

# Each slice uses the same subsampling pattern
ksp_full_ms = rand(ComplexF32, nx, ny, 8, nz)
ksp_sub_ms = similar(ksp_full_ms, sum(mask_2d), 8, nz)
for iz in 1:nz
    ksp_sub_ms[:, :, iz] = ksp_full_ms[indices_2d, :, iz]
end

println("Multi-slice subsampled size: ", size(ksp_sub_ms))
```

### Supported Pattern Formats

**2D Patterns:**

| Format | Example | Description |
|--------|---------|-------------|
| Boolean 2D | `mask::Array{Bool,2}` | Arbitrary 2D sampling |
| Linear indices | `indices::Vector{Int}` | Flattened k-space indices |
| Cartesian indices | `indices::Vector{CartesianIndex{2}}` | 2D coordinates |
| Separable | `(pattern_x, pattern_y)` | Independent per dimension |

**3D Patterns:**

| Format | Example | Description |
|--------|---------|-------------|
| Boolean 3D | `mask::Array{Bool,3}` | Arbitrary 3D sampling |
| Separable | `(pattern_x, pattern_y, pattern_z)` | Independent per dimension |
| Hybrid 1D+2D | `(pattern_x, mask_yz)` | 1D pattern + 2D mask |
| Hybrid 2D+1D | `(mask_2d, pattern_z)` | 2D mask + 1D pattern |

### Acceleration Factor

The acceleration factor $R$ quantifies undersampling:

```@example ops
mask = rand(Bool, 128, 128)
full_samples = 128 * 128
acquired_samples = sum(mask)
acceleration = full_samples / acquired_samples
println("Acceleration factor: ", round(acceleration, digits=2), "x")
```

**Typical ranges:**
- Cartesian 2D: R = 2-6
- Cartesian 3D: R = 3-10
- Radial/Spiral: R = 4-12
- Random: R = 2-8

Higher acceleration requires stronger regularization for good reconstruction quality.

## Integration with AcquisitionInfo

Extract operators directly from `AcquisitionInfo`:

```@example ops
# Create acquisition info
mask_acq = rand(Bool, 64, 64)
indices_acq = findall(mask_acq)
ksp_sub_acq = rand(ComplexF32, sum(mask_acq), 8)
smaps_acq = coil_sensitivities(64, 64, 8)

acq_info = AcquisitionInfo(
    ksp_sub_acq; 
    is3D=false,
    image_size=(64, 64),
    subsampling=(mask_acq,),
    sensitivity_maps=smaps_acq
)

# Extract operators
E = get_encoding_operator(acq_info)
S = get_sensitivity_map_operator(acq_info)
𝒫 = get_subsampling_operator(acq_info)

println("Encoding operator type: ", typeof(E))
```

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

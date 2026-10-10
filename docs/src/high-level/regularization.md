# Regularization

*Tutorial: [Regularization](../tutorials/05_regularization.md).*

Regularization is essential for reconstructing high-quality images from undersampled k-space data. This page explains the available regularization methods and how to use them.

## Why Regularization?

When k-space is undersampled (as in compressed sensing or parallel imaging), the reconstruction problem becomes **ill-posed** - there are many possible images that could have produced the observed data. Regularization adds prior knowledge about what "good" images look like to guide the reconstruction toward a unique, high-quality solution.

## Understanding the Math

For those interested in the mathematical details, the reconstruction solves:

```
minimize  (1/2)‖E·x - y‖₂² + ∑ᵢ λᵢ·Rᵢ(x)
```

Where:
- `E` is the encoding operator (Fourier + sensitivity + subsampling)
- `x` is the image to reconstruct
- `y` is the observed k-space data
- `‖E·x - y‖₂²` is the data fidelity term
- `Rᵢ(x)` are the regularization terms
- `λᵢ` are the regularization parameters

The first term ensures the reconstruction is consistent with observed data. The regularization terms encode prior knowledge about image properties.

## Available Regularization Methods

```@setup imports
using Ristretto
using MIRTjim: jim
using Plots
using Random

Random.seed!(0)
```

The code snippets in the following sections assume that `Ristretto` and `MIRTjim` are already imported. `MIRTjim` is a convenience wrapper around `Plots.jl` for displaying multidimensional images. Also, assume you have an `AcquisitionInfo` object `acq` representing your k-space data and acquisition settings for simulated Shepp-Logan phantom:

```@example imports
using Ristretto
using GeometricMedicalPhantoms
using MIRTjim: jim
using Ristretto: get_operator

# Simulate 2D acquisition
x = create_shepp_logan_phantom(128, 128, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
x_noisy = x + 0.02f0 * randn(ComplexF32, 128, 128)
smaps = coil_sensitivities(128, 128, 8)
pdf = VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05)
pattern = create_sampling_pattern(pdf, (128, 128))
acq_full = AcquisitionInfo(
    is3D=false, 
    image_size=(128, 128), 
    subsampling=pattern, 
    sensitivity_maps=smaps
)
data = simulate_acquisition(x_noisy, acq_full; inverse_crime_check = false, keep_sensitivity_maps = true)

# Simulate 3D acquisition
x3d = create_shepp_logan_phantom(64, 64, 32; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps3d = coil_sensitivities(64, 64, 32, 8)
subsampling3d = create_sampling_pattern(
    VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05), 
    (64, 64, 32)
)
acq3d = AcquisitionInfo(
    image_size=(64, 64, 32), 
    sensitivity_maps=smaps3d,
    subsampling=subsampling3d,
)
data3d = simulate_acquisition(x3d, acq3d; inverse_crime_check = false, keep_sensitivity_maps = true)

jim(x3d; title="Shepp-Logan Phantom (3D)", size=(800,400))
savefig("shepp_logan_phantom_3d.png"); nothing # hide
```

![shepp_logan_phantom_3d.png](shepp_logan_phantom_3d.png)

### Image Domain Regularization

#### ℓ₂ Image Domain (Tikhonov) Regularization

The simplest form of regularization, penalizing large pixel values. `L2Image` is the canonical name;
`Tikhonov` is an exported alias for the same type.

```@docs
L2Image
```

**When to use:**
- Noise reduction without strong assumptions about image structure
- As a baseline for comparison with other methods, especially for parallel imaging
- Fast and simple regularization (it is computationally cheap and can be optimized with Conjugate Gradient)

**Example:**
```@example imports
img₁ = reconstruct(data, IterativeReconstruction(L2Image(1e-1)); verbosity = Silent())
img₂ = reconstruct(data, IterativeReconstruction(L2Image(1e-6)); verbosity = Silent())
p1 = jim(img₁; title="L2Image λ=1e-1")
p2 = jim(img₂; title="L2Image λ=1e-6")
jim(p1, p2; layout=(1,2), size=(800,400))
savefig("tikhonov_regularization.png"); nothing # hide
```

![tikhonov_regularization.png](tikhonov_regularization.png)

#### L1 Regularization

Promotes sparse images (many pixels close to zero):

```@docs
L1Image
```

**When to use:**
- Images that are naturally sparse (e.g., angiography) and you want to suppress small values

**Example:**
```@example imports
img₁ = reconstruct(data, IterativeReconstruction(L1Image(1e-2)); verbosity = Silent())
img₂ = reconstruct(data, IterativeReconstruction(L1Image(1e-5)); verbosity = Silent())
p1 = jim(img₁; title="L1Image λ=1e-2")
p2 = jim(img₂; title="L1Image λ=1e-5")
jim(p1, p2; layout=(1,2), size=(800,400))
savefig("l1image_regularization.png"); nothing # hide
```

![l1image_regularization.png](l1image_regularization.png)

### Wavelet Domain Regularization

#### 2D Wavelet Sparsity

Promotes sparsity in the wavelet domain:

```@docs
L1Wavelet2D
```

**When to use:**
- Natural images with structure at multiple scales
- Most MRI applications (anatomy has multi-scale features)
- Standard compressed sensing reconstruction

**Parameters:**
- `λ`: Regularization strength (try 1e-3 to 1e-2)
- `wavelet`: Wavelet type (default: Daubechies, also try Haar, etc.)
- `levels`: Number of decomposition levels (default: 4)

**Example:**
```@example imports
reg = L1Wavelet2D(1e-3)
example_img = rand(ComplexF32, 128, 128)
op = get_operator(reg, example_img)
transformed = op * x_noisy
p1 = jim(transformed; title="Wavelet Coefficients")
img = reconstruct(data, IterativeReconstruction(reg); verbosity = Silent())
p2 = jim(img; title="L1Wavelet2D Reconstruction")
jim(p1, p2; layout=(1,2), size=(800,400))
savefig("l1wavelet2d_regularization.png"); nothing # hide
```

![l1wavelet2d_regularization.png](l1wavelet2d_regularization.png)

**Options for `L1Wavelet2D`:**
- `wavelet`: Specify wavelet type (e.g., `WT.haar`, `WT.db4`)
- `levels`: Number of decomposition levels (default: 4)

```@example imports
reg_haar = L1Wavelet2D(1e-2; wavelet=WT.haar)
op_haar = get_operator(reg_haar, example_img)
transformed_haar = op_haar * x_noisy
img_haar = reconstruct(data, IterativeReconstruction(reg_haar); verbosity = Silent())
p1 = jim(transformed_haar; title="Haar Coefficients")
p2 = jim(img_haar; title="Haar Reconstruction")

reg_level8 = L1Wavelet2D(1e-3; levels=8)
op_level8 = get_operator(reg_level8, example_img)
transformed_level8 = op_level8 * x_noisy
img_level8 = reconstruct(data, IterativeReconstruction(reg_level8); verbosity = Silent())
p3 = jim(transformed_level8; title="Level 8 Coefficients")
p4 = jim(img_level8; title="Level 8 Reconstruction")
jim(p1, p2, p3, p4; layout=(2,2), size=(800,700))
savefig("l1wavelet2d_options.png"); nothing # hide
```

### Contourlet Domain Regularization

#### Contourlet (NSCT) Sparsity

Promotes sparsity in the Nonsubsampled Contourlet Transform (NSCT) domain -- unlike wavelets,
contourlets capture directional/curve-like structure (edges, vessels) with fewer coefficients:

```@docs
L1Contourlet
```

**When to use:**
- Images dominated by directional edges or elongated structures (vasculature, fibrous tissue)
- As an alternative to [`L1Wavelet2D`](@ref) when wavelet's isotropic basis under-represents oriented features

**Parameters:**
- `λ`: Regularization strength (scalar only, try 1e-3 to 1e-2)
- `params`: `ContourletParams` controlling pyramid levels/directions (default: `J=3`, `parabolic_levels(3)`)

**Example:**
```@example imports
reg = L1Contourlet(1e-3)
op = get_operator(reg, example_img)
transformed = op * x_noisy
p1 = jim(transformed[:, :, 1]; title="Contourlet Coarse Band")
img = reconstruct(data, IterativeReconstruction(reg); verbosity = Silent())
p2 = jim(img; title="Contourlet Reconstruction")
jim(p1, p2; layout=(1,2), size=(800,400))
savefig("contourlet_regularization.png"); nothing # hide
```

![contourlet_regularization.png](contourlet_regularization.png)

![l1wavelet2d_options.png](l1wavelet2d_options.png)

#### 3D Wavelet Sparsity

For volumetric / multislice data, promotes sparsity in 3D wavelet domain:

```@docs
L1Wavelet3D
```

**When to use:**
- 3D acquisitions or multi-slice 2D data
- When you want to exploit 3D structure

**Example:**
```@example imports
reg3d = L1Wavelet3D(1e-3)

op = get_operator(reg3d, rand(ComplexF32, 64, 64, 32))
transformed = op * x3d
jim(abs.(transformed); title="3D Wavelet Coefficients", size=(800,400))
savefig("l1wavelet3d_coefficients.png"); nothing # hide
```

![l1wavelet3d_coefficients.png](l1wavelet3d_coefficients.png)

### Total Variation

#### 2D Total Variation

Promotes piecewise-constant images by penalizing rapid changes:

```@docs
TotalVariation2D
```

**When to use:**
- Images with sharp edges and flat regions
- Brain imaging with gray/white matter boundaries
- When you want strong edge preservation

**Example:**
```@example imports
reg = TotalVariation2D(1e-3)
op = get_operator(reg, example_img)
transformed = op * x_noisy
img = reconstruct(data, IterativeReconstruction(reg); verbosity = Silent())
p1 = jim(transformed[:,:,1]; title="Δx Coefficients")
p2 = jim(transformed[:,:,2]; title="Δy Coefficients")
p3 = jim(img; title="TotalVariation2D Reconstruction")
jim(p1, p2, p3; layout=(1, 3), size=(900,250))
savefig("totalvariation2d_coefficients.png"); nothing # hide
```

![totalvariation2d_coefficients.png](totalvariation2d_coefficients.png)

**Practical tip:** TV can create a "cartoon-like" appearance. Use lower λ values (1e-4 to 5e-3) to preserve texture.

#### 3D Total Variation

For volumetric / multislice data, promotes piecewise-constant structure in 3D:

```@docs
TotalVariation3D
```

#### Anisotropic Total Variation

The same finite differences, summed with an ℓ₁ norm instead of the isotropic ℓ₂,₁ mixed norm — the
directional derivatives are penalized independently rather than as a gradient vector per pixel:

```@docs
AnisotropicTotalVariation2D
AnisotropicTotalVariation3D
```

**When to use:**
- Structure that is genuinely axis-aligned (phantoms, grids, rectangular hardware)
- When the separable prox matters: it is plain soft thresholding of the difference coefficients

**The trade-off:** the penalty is not rotation invariant. It is cheapest for horizontal and vertical
edges, so diagonal boundaries can pick up a faint staircase texture that
[`TotalVariation2D`](@ref) does not produce. Isotropic TV remains the default for anatomy.

```@example imports
img_aniso = reconstruct(data, IterativeReconstruction(AnisotropicTotalVariation2D(1e-3)); verbosity = Silent())
img_iso = reconstruct(data, IterativeReconstruction(TotalVariation2D(1e-3)); verbosity = Silent())
jim(
    jim(img_iso; title="isotropic TV"),
    jim(img_aniso; title="anisotropic TV");
    layout=(1, 2), size=(600,250)
)
savefig("anisotropic_tv.png"); nothing # hide
```

![anisotropic_tv.png](anisotropic_tv.png)

#### Second-Order Total Variation

Penalizes the second derivatives instead of the first, so a smooth intensity ramp costs nothing:

```@docs
SecondOrderTotalVariation2D
SecondOrderTotalVariation3D
```

**When to use:**
- Images dominated by smooth intensity variation (coil shading, slow tissue transitions, B1 inhomogeneity), where first-order TV produces staircasing — flat plateaus separated by artificial steps
- Almost always in combination with a first-order term rather than alone, since a jump is penalized through its (large) second derivative and is therefore blurred

**Example:**
```julia
# first-order TV for the edges, second-order for the ramps
img = reconstruct(acq, IterativeReconstruction(TotalVariation2D(1e-3), SecondOrderTotalVariation2D(2e-3)))
```

**Practical tip:** If you find yourself tuning the balance between first- and second-order TV, use
[`TotalGeneralizedVariation2D`](@ref) instead: it makes the same trade-off adaptively, per voxel.

#### Total Generalized Variation

Balances first- and second-order behaviour automatically through an auxiliary vector field:

```@docs
TotalGeneralizedVariation2D
TotalGeneralizedVariation3D
```

**When to use:**
- The default replacement for [`TotalVariation2D`](@ref) whenever staircasing is a concern, i.e. on any image that is not genuinely piecewise constant — which in MRI is most of them
- Especially worthwhile at high acceleration, where the TV staircasing artifact is strongest
- Use [`TotalGeneralizedVariation3D`](@ref) for volumetric data, so that the auxiliary field and the
  symmetrized gradient run over all three spatial dimensions instead of treating slices independently

**Example:**
```julia
img = reconstruct(acq, IterativeReconstruction(TotalGeneralizedVariation2D(1e-3); algorithm = ADMM(), maxit = 500))
```

**Practical tip:** TGV requires `ADMM` — the auxiliary field is coupled to the image through `∇x − w`, which
the proximal-gradient algorithms cannot separate. It also doubles the number of unknowns, so expect roughly
twice the memory and a modest increase in cost per iteration. `ratio` rarely needs changing from its default
of `2.0`; increasing it makes the result approach plain TV.

#### Infimal-Convolution Total Variation

The infimal convolution of first- and second-order TV needs no regularization type of its own: it *is* an
image decomposition into a piecewise-constant "cartoon" part and a piecewise-linear "ramp" part, which the
[Image Decomposition](image_decomposition.md) machinery already expresses.

```julia
components = (
    Component(:cartoon, TotalVariation2D(1e-3)),
    Component(:ramp, SecondOrderTotalVariation2D(1e-3)),
)
img = reconstruct(acq, IterativeReconstruction(components...; algorithm = ADMM(), maxit = 500))
img.cartoon   # the edges, shorthand for img.components.cartoon
img.ramp      # the smooth background, shorthand for img.components.ramp
```

Compared to [`TotalGeneralizedVariation2D`](@ref) this is the older and slightly weaker model — the two parts
are separated globally rather than adaptively per voxel — but it has the advantage that the separated
components are themselves available, which is useful when the smooth part is the bias field or the background.

#### Edge-Preserving Roughness (Huber)

A smooth interpolation between a quadratic roughness penalty and total variation:

```@docs
EdgePreservingRoughness2D
EdgePreservingRoughness3D
```

**When to use:**
- When TV's piecewise-constant bias is unwanted but a quadratic penalty over-smooths edges
- With gradient-based algorithms: the penalty is differentiable everywhere, so it needs no proximal step
- Statistical / model-based reconstruction, where this is the classical choice of potential function

**Example:**
```julia
img = reconstruct(acq, IterativeReconstruction(EdgePreservingRoughness2D(1e-3; δ = 0.01)))
```

**Practical tip:** `δ` is an absolute intensity, so it must be set relative to the image scale. A workable
recipe is to take a preliminary reconstruction, compute the magnitudes of its finite differences, and use a
low percentile (5–20%) of those as `δ`: differences below that count as noise and are smoothed quadratically,
those above count as edges and are preserved.

### Temporal Regularization

#### Temporal Fourier Sparsity

For dynamic imaging, promotes sparsity in the temporal Fourier domain:

```@docs
L1TemporalFourier
```

**When to use:**
- Dynamic or cine imaging
- Cardiac MRI
- DCE-MRI (dynamic contrast enhanced)
- When motion is periodic or smoothly varying

**Example:**
```julia
img = reconstruct(acq, IterativeReconstruction(L1TemporalFourier(1e-2, time_dim=4)))
```

**Practical tip:** This works best when temporal changes are smooth or periodic. For irregular motion, consider temporal total variation or low-rank methods instead.

#### Temporal Total Variation

For dynamic imaging with irregular or non-periodic motion, penalizes the frame-to-frame differences:

```@docs
TemporalTotalVariation
```

**When to use:**
- Free-breathing and real-time acquisitions, where the temporal Fourier assumption of periodicity fails
- Contrast dynamics that are piecewise smooth in time (DCE-MRI, first-pass perfusion)
- As the sparse part of an L+S model (see [Image Decomposition](image_decomposition.md))

**Example:**
```julia
img = reconstruct(acq_dynamic, IterativeReconstruction(TemporalTotalVariation(2e-2; time_dim = 3)))
```

**Practical tip:** Like spatial TV, this term uses a non-tight operator, so reconstruction falls back to ADMM. It is the temporal counterpart of [`TotalVariation2D`](@ref) and is often combined with it (`(TotalVariation2D(1e-3), TemporalTotalVariation(2e-2))`) — the "spatiotemporal TV" of the golden-angle radial sparse parallel (GRASP) literature.

### Low-Rank Regularization

#### Nuclear Norm

Promotes low-rank structure in dynamic data:

```@docs
LowRank
RankLimit
```

**When to use:**
- Dynamic imaging with temporal correlations
- Background suppression in DCE-MRI
- Data with strong spatiotemporal correlations
- When images share common features across time

**Example:**
```julia
# Dynamic series with low-rank structure
img = reconstruct(acq_dynamic, IterativeReconstruction(LowRank(1e-1)))
```

**Practical tip:** Low-rank methods can be computationally expensive. Use for datasets where temporal correlations are strong.

#### Locally Low Rank

Instead of one Casorati matrix for the whole image, penalizes the nuclear norm of every spatial block separately:

```@docs
LocallyLowRank
```

**When to use:**
- Dynamic series where the temporal dynamics differ across the field of view (cardiac motion vs. static background, focal contrast uptake) — a global low-rank model needs a high rank to represent all of them at once, a local one does not
- Quantitative parameter mapping (T1/T2 relaxometry, MR fingerprinting), where each voxel neighbourhood follows a low-dimensional signal model

**Example:**
```julia
img = reconstruct(acq_dynamic, IterativeReconstruction(LocallyLowRank(5e-2; block_size = 8, time_dim = 3)))
```

**Practical tip:** `block_size` trades locality against cost and stability: 4-8 voxels for strongly varying dynamics, 12-16 when the temporal signal is smooth over larger regions. Each iteration performs one SVD of a `(∏ block_size) × n_frames` matrix per block. A single fixed block grid can leave visible block boundaries at large λ; pass `shift = :random` to redraw the grid before every proximal step, which averages them out (see below).

##### Shifting the Block Grid

`LocallyLowRank` accepts a `shift` argument controlling where the tiling grid starts:

| `shift` | Grid | Objective | Use with |
|---|---|---|---|
| `:none` (default) | fixed at the first voxel | stationary | any algorithm |
| `:fixed` | one random origin, drawn once | stationary | any algorithm |
| `:random` | redrawn before every prox | changes per iteration | `ISTA`, `FISTA`, `ADMM` only |

`:random` is the standard remedy for block artifacts in the literature. The grid wraps circularly, so the
tiling remains a permutation of the voxels and the prox stays exact — but only when every spatial extent is
divisible by the block edge, which is checked. Because the objective is no longer the same function at every
iteration, the line-search algorithms (`PANOC`, `PANOCplus`, `ZeroFPR`) must not be used with it.

```julia
img = reconstruct(acq_dynamic, IterativeReconstruction(LocallyLowRank(5e-2; block_size = 8, time_dim = 3, shift = :random); algorithm = FISTA()))
```

#### Multi-Scale Low Rank

Penalizes the same block-wise nuclear norm at several block sizes at once:

```@docs
MultiScaleLowRank
```

**When to use:**
- Dynamic series containing both large, globally correlated dynamics (respiratory motion of the whole field of view) and small, localized ones (focal contrast uptake), where no single `block_size` is right for both
- As a less sensitive alternative to tuning `block_size` for [`LocallyLowRank`](@ref)

**Example:**
```julia
img = reconstruct(acq_dynamic, IterativeReconstruction(MultiScaleLowRank(5e-2; block_sizes = (4, 8, 16), time_dim = 3)))
```

**Practical tip:** The term uses the *proximal average* of the per-scale penalties, which approximates their
sum; the objective value it reports is the weighted average of the per-scale penalties. If you want the exact
multi-scale model of Ong & Lustig — one separate image component per scale — build it from components instead,
which also gives you the separated scales:

```julia
components = Tuple(
    Component(Symbol(:scale, b), LocallyLowRank(5e-2; block_size = b, time_dim = 3)) for b in (4, 8, 16)
)
```

Cost grows linearly with the number of scales, so two or three are usually enough.

`λ` may also be a vector with one entry per scale, giving coarse and fine scales independent
thresholds instead of the single value applied everywhere:

```julia
MultiScaleLowRank([2e-2, 5e-2, 1e-1]; block_sizes = (4, 8, 16), time_dim = 3)
```

This is a different knob from `weights`: `λⱼ` is scale `j`'s own nuclear-norm threshold, while
`weights` is the convex combination the proximal average takes across the (already-thresholded)
scales. Doubling `weights[j]` changes how much scale `j`'s penalty counts toward the reported
average; doubling `λⱼ` changes what scale `j`'s prox actually shrinks toward.

#### Structured Low Rank k-Space (SAKE / LORAKS-C)

Promotes low rank of the **block-Hankel matrix** built from sliding windows over multi-coil
k-space, with the coils stacked as extra columns. This is a *calibrationless* parallel-imaging
prior: it recovers the missing samples of an undersampled multi-coil acquisition without
sensitivity maps and without an ACS-calibrated kernel. The structure it exploits is the same
linear predictability GRAPPA and SPIRiT use, except that it is estimated from the undersampled
data itself.

```@docs
StructuredLowRank
```

**When to use:**
- Undersampled multi-coil Cartesian data with **no** sensitivity maps and **no** (or too small) calibration region
- As a complement to, or replacement for, GRAPPA/SPIRiT when calibration lines are unavailable
- Moderate acceleration (R ≈ 2-4); higher factors need a calibration region or a warm-started `x₀`

**Parameters:**
- Exactly one of `λ` (nuclear-norm penalty, the convex LORAKS-C form) or `max_rank` (hard rank
  constraint, the SAKE form) — they are mutually exclusive, as for [`L0Image`](@ref).
- `window` is the sliding-window size over the k-space encoding dimensions, typically `(5, 5)` or `(6, 6)` in 2D and `(4, 4, 4)` in 3D.

**Example:**
```julia
# calibrationless: acq holds multi-coil k-space, no sensitivity_maps
rec = reconstruct(
    acq,
    IterativeReconstruction(
        StructuredLowRank(; λ = 0.03, window = (6, 6));
        signal_model = KSpaceToImage(RootSumSquares()),
        algorithm = ADMM(),
        maxit = 100,
    ),
)
```

!!! warning "The `max_rank` form is non-convex"
    A hard rank cap is a projection onto a non-convex set, and it is applied to the *lifted*
    matrix rather than to k-space itself, so any splitting algorithm using it is a heuristic:
    the result depends on the starting estimate and convergence is not guaranteed. Use the `λ`
    form when you want a convex problem.

**Practical tip:** The optimization variable is the full multi-channel k-space, so pair the term
with `signal_model = KSpaceToImage(...)` (as `SPIRiT(; iterative = true)` does). Each iteration
costs one economy SVD of a `prod(gridsize .- window .+ 1) × (prod(window) * ncoils)` matrix per
batch slab, so keep `window` small.

**LORAKS phase constraints — the S and G matrices.** `structure` selects which LORAKS matrix is
lifted. `:c` (the default) is the block-Hankel matrix above, low rank when the image has limited
support and — with several coils — through the coil relations. `:s` is the LORAKS S-matrix, which
reads k-space on both sides of DC and is low rank when the image **phase varies smoothly**; `:g`
is the other phase construction of Haldar (2014), offered for completeness and weaker by the
paper's own analysis (rank-deficient, but not necessarily *low* rank unless the support is
limited too). Both are real by construction, so their prox runs a real SVD of a matrix with twice
the rows and columns of the C matrix: 2.7-2.8× the C-matrix cost per proximal call, measured on a
64²×8 slab with a `(5, 5)` window (76 ms against 209 ms and 212 ms).

Unlike `:c`, the phase structures say something about a **single-channel** acquisition, which is
LORAKS' original point: no coils, no calibration, no phase estimate. On a 32×32 single-channel
partial-Fourier problem (62 % of `ky` from one side) they cut the NRMSE from 0.182 zero-filled to
0.123 (`:s`) and 0.132 (`:g`).

```julia
StructuredLowRank(; λ = 0.02, window = (5, 5), structure = :s)
```

The reflection about DC uses Ristretto's centered k-space convention by default; pass `kspace_center`
when DC sits elsewhere (an acquisition with `shifted_kspace_dims` has it at index 1). ALOHA's
`weights` apply to `:c` only. To impose support *and* phase structure at once, as LORAKS does,
add two terms — one `:c`, one `:s` — each with its own `λ`.

**ALOHA — transform-domain weighting.** `weights` lifts `w ⊙ k` instead of `k`, where `w` is the
annihilating filter implied by a sparsity model: if a transform of the image is sparse, the
correspondingly weighted k-space is annihilated by a short filter, so its block-Hankel matrix is
low-rank *even for a single channel* — a structure on top of the coil relations SAKE and LORAKS-C
use. `:tv` is one first-difference weight per encoding dimension; `:wavelet` adds the step-2 Haar
detail band; an array (or a collection of arrays) is a custom weight or pyramid. Several weights
are combined by a proximal average, as [`MultiScaleLowRank`](@ref) combines its scales, and the
first entry of a built-in model is the approximation band — the plain unweighted term — because a
difference weight vanishes at DC and the detail bands alone constrain nothing there.

```julia
StructuredLowRank(; λ = 0.03, window = (6, 6), weights = :tv)
```

!!! note "When the weighting pays"
    The annihilation only holds when the sparsifying transform's support is small enough for the
    window: a filter of edge length `w` annihilates roughly `w - 1` spikes per direction. On the
    package's toy phantoms — whose gradient support is hundreds of pixels at `window = (5, 5)` —
    the weighted terms measurably *worsen* the reconstruction relative to plain LORAKS-C, because
    a lift that is not low-rank is being pushed toward low rank anyway. Verify on your own data,
    with `weights = nothing` as the baseline, before adopting it.

### Hard Thresholding

Penalizes or constrains the *number* of non-zero coefficients rather than their magnitude. One type per
sparsifying transform, each taking `threshold` (penalty form) XOR `count` (constraint form):

```@docs
L0Image
L0Wavelet2D
L0Wavelet3D
```

**When to use:**
- When the amplitude bias of the ℓ₁ terms is a problem: soft thresholding shrinks the coefficients it keeps, hard thresholding does not, so lesion or vessel intensities are not systematically underestimated
- The `count` form when the sparsity level is known a priori and is easier to specify than a penalty weight — the same argument that makes [`RankLimit`](@ref) preferable to [`LowRank`](@ref) in some settings

**Example:**
```julia
# ℓ₀ penalty on wavelet coefficients, warm-started from an ℓ₁ solution
x_l1 = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(1e-3)))
img = reconstruct(acq, IterativeReconstruction(L0Wavelet2D(threshold = 1e-3)); x₀ = x_l1)
```

**Practical tip:** Both terms are non-convex, so the solvers only guarantee a stationary point and the result
depends on the starting image. Warm-starting from an ℓ₁ reconstruction is the reliable recipe. Note also that
the threshold is `sqrt(2γλ)` rather than `γλ`, so a `λ` carried over from an ℓ₁ term will not give a
comparable sparsity level.

### Plug-and-Play Priors

Uses an off-the-shelf image denoiser as the proximal operator, i.e. as an implicit image prior:

```@docs
PlugAndPlay
```

**When to use:**
- When a denoiser is available that encodes far more about the images than any hand-written penalty — a learned denoiser (DnCNN and successors) or BM3D
- As a drop-in upgrade of a wavelet or TV term without changing the reconstruction pipeline

**Example:**
```julia
using BM3D
img = reconstruct(
    acq,
    IterativeReconstruction(
        PlugAndPlay((image, σ) -> bm3d(image, σ); strength = 0.05);
        algorithm = FISTA(),
        maxit = 100,
    ),
)
```

**Practical tip:** No denoiser ships with this package; anything callable as `denoiser(image, σ)` works. The
implicit prior has no value function, so the reported objective is `NaN` and objective-based convergence
checks are meaningless — use `ISTA`, `FISTA` or `ADMM` with a fixed iteration budget, and never the
line-search algorithms. To check the wiring end to end, a soft-thresholding "denoiser" reproduces
[`L1Image`](@ref) exactly.

### Joint Sparsity

For multi-contrast, multi-echo or multi-directional data, forces the components to share a common support:

```@docs
JointSparsity
```

**When to use:**
- Multi-echo, multi-contrast (T1w/T2w/FLAIR) or diffusion data of the same anatomy: the edges are in the same place in every image, only their intensities differ
- Velocity- or phase-encoded series
- Preferable to independent `L1Image`/`L1Wavelet2D` on each contrast, because the joint norm couples them

**Example:**
```julia
# echoes stored along dimension 3, sharing the same support
img = reconstruct(acq_multiecho, IterativeReconstruction(JointSparsity(1e-2; dim = 3)))
```

**Practical tip:** Joint sparsity is most effective on a sparsifying transform of the images. Combining `JointSparsity` with a wavelet regularizer per contrast (`(JointSparsity(1e-2; dim = 3), L1Wavelet2D(1e-3))`) is a common compromise.

### Reference-Image Prior

Promotes sparsity of the *difference* to a known image instead of the image itself:

```@docs
ReferencePrior
```

**When to use:**
- Dynamic series where a high-quality temporal average or a previous time frame is available
- Follow-up or multi-contrast exams where an earlier high-SNR scan of the same anatomy exists
- Interventional / real-time imaging with a fully sampled baseline

**Example:**
```julia
x_ref = reconstruct(acq_reference, IterativeReconstruction(L1Wavelet2D(1e-3)))
img = reconstruct(acq, IterativeReconstruction(ReferencePrior(1e-2, x_ref), L1Wavelet2D(1e-3)))
```

**Practical tip:** The reference must be in the same units as the reconstruction; when data scaling is enabled the reference is rescaled automatically. A wrong reference biases the result toward it, so combine it with an ordinary sparsity term (as in the PICCS convex combination) rather than using it alone.

### Constraints

Constraints are enforced exactly by projection instead of being traded off against data consistency, so they carry no `λ`:

```@docs
NonNegative
BoxConstraint
```

**When to use:**
- Quantitative maps with a physically meaningful range (proton density, relaxation rates, diffusion coefficients)
- Magnitude-only or phase-resolved real-valued reconstructions

By default, applying either to complex data throws an `ArgumentError` — the plain non-negative
orthant and box are only ordered for real numbers. Pass `complex_handling = :real` to project a
complex image onto the real orthant/box instead (imaginary part discarded, real part clamped),
following RegularizedLeastSquares.jl's `PositiveRegularization` convention.

**Elsewhere:** BART exposes the non-negative constraint as `pics -R S:0:0:0`, and constrains the
image to be real-valued with the separate `pics -c` flag. SigPy has
`sigpy.prox.BoxConstraint(shape, lower, upper)`, with `lower = 0` for non-negativity.
RegularizedLeastSquares.jl (used by MRIReco.jl) has `PositiveRegularization` and
`RealRegularization`, which take the real part the way `complex_handling = :real` does.

**Example:**
```julia
# real-valued image (e.g. a magnitude reconstruction)
img = reconstruct(acq_real, IterativeReconstruction(TotalVariation2D(1e-3), NonNegative()))

# complex image, projected onto the real non-negative orthant each iteration
img = reconstruct(
    acq, IterativeReconstruction(TotalVariation2D(1e-3), NonNegative(; complex_handling = :real))
)
```

## Combining Multiple Regularizers

You can combine multiple regularization terms to exploit different image properties simultaneously:

```julia
# Comprehensive regularization for dynamic imaging
method = IterativeReconstruction(
    L1Wavelet2D(5e-3),      # Spatial sparsity
    TotalVariation2D(1e-3),  # Edge preservation
    L1TemporalFourier(2e-2),   # Temporal smoothness
)
img = reconstruct(acq_dynamic, method)
```

**When to combine:**
- Wavelet + TV: Exploit both multi-scale structure and edge preservation
- Spatial + Temporal: Regularize both space and time dimensions
- Multiple spatial regularizers: When images have complex structure

## Choosing Regularization Parameters

The regularization parameter λ controls the trade-off between data fidelity and regularization:

- **Too small (λ → 0)**: Noisy, artifacts remain
- **Too large (λ → ∞)**: Over-smoothed, loss of detail
- **Just right**: Balance between noise/artifact suppression and detail preservation

### Practical Guidelines

**Starting values by regularization type:**
- L2Image: `1e-5` to `1e-3`
- L1Image: `1e-4` to `1e-2`
- L1Wavelet: `1e-3` to `1e-2`
- TotalVariation: `1e-4` to `5e-3`
- AnisotropicTotalVariation: `1e-4` to `5e-3`, the same range as the isotropic term
- SecondOrderTotalVariation: `1e-4` to `1e-2` (roughly 2× the first-order λ when the two are combined)
- TotalGeneralizedVariation2D/3D: `1e-4` to `5e-3`, i.e. the same range as `TotalVariation2D`; leave `ratio` at `2.0`
- EdgePreservingRoughness: `1e-4` to `5e-3` for λ; `δ` from the gradient magnitudes of a preliminary reconstruction
- L1TemporalFourier: `1e-2` to `1e-1`
- TemporalTotalVariation: `1e-2` to `1e-1`
- LowRank: `1e-2` to `1`
- LocallyLowRank: `1e-2` to `5e-1`
- MultiScaleLowRank: `1e-2` to `5e-1`, as for LocallyLowRank
- StructuredLowRank, `λ` form: `1e-2` to `1e-1`, relative to the k-space scale; `max_rank` form: no λ, set the rank from the singular-value spectrum of `𝓗 k` on the zero-filled data
- L0Image/L0Wavelet2D/L0Wavelet3D, `threshold` form: `1e-4` to `1e-2`, but note the `sqrt(2γλ)` threshold — retune rather than reusing an ℓ₁ λ
- L0Image/L0Wavelet2D/L0Wavelet3D, `count` form: no λ; set the coefficient budget from the expected sparsity
- PlugAndPlay: `strength` `1e-2` to `1e-1`, in the units of the image intensity
- JointSparsity: `1e-3` to `1e-2`
- ReferencePrior: `1e-3` to `1e-1`
- NonNegative / BoxConstraint: no parameter

**Adjustment strategy:**
1. Start with the suggested value
2. If too noisy/aliased → increase λ
3. If too smooth/blurry → decrease λ
4. Typical range: adjust by factors of 2-5

## Choosing a Regularizer

| Data | First choice | Common combinations |
|---|---|---|
| Static 2D/3D anatomy | [`L1Wavelet2D`](@ref) / [`L1Wavelet3D`](@ref) | + [`TotalVariation2D`](@ref) |
| Piecewise-constant anatomy, strong edges | [`TotalVariation2D`](@ref) / [`TotalVariation3D`](@ref) | + [`L1Wavelet2D`](@ref) |
| Edges *and* smooth intensity variation (staircasing is a problem) | [`TotalGeneralizedVariation2D`](@ref) | infimal convolution: [`TotalVariation2D`](@ref) + [`SecondOrderTotalVariation2D`](@ref) as components |
| Smooth penalty wanted (gradient-based solver, model-based recon) | [`EdgePreservingRoughness2D`](@ref) | + [`L1Wavelet2D`](@ref) |
| ℓ₁ amplitude bias is a problem | [`L0Image`](@ref) / [`L0Wavelet2D`](@ref) / [`L0Wavelet3D`](@ref) | warm-started from an ℓ₁ solution |
| A trained or off-the-shelf denoiser is available | [`PlugAndPlay`](@ref) | — |
| Periodic dynamics (cine, cardiac) | [`L1TemporalFourier`](@ref) | + [`TotalVariation2D`](@ref) |
| Irregular dynamics (free-breathing, real-time) | [`TemporalTotalVariation`](@ref) | + [`TotalVariation2D`](@ref) |
| Strong global spatiotemporal correlation (DCE, perfusion) | [`LowRank`](@ref) | L+S: [`LowRank`](@ref) + [`TemporalTotalVariation`](@ref), see [Image Decomposition](image_decomposition.md) |
| Spatially varying dynamics, parameter mapping | [`LocallyLowRank`](@ref) | + [`TotalVariation2D`](@ref) |
| Dynamics at several spatial scales at once | [`MultiScaleLowRank`](@ref) | one [`LocallyLowRank`](@ref) component per scale |
| Multi-contrast / multi-echo / diffusion | [`JointSparsity`](@ref) | + [`L1Wavelet2D`](@ref) |
| A high-quality prior image exists | [`ReferencePrior`](@ref) | + [`L1Wavelet2D`](@ref) |
| Real-valued images, physical range known | [`NonNegative`](@ref) / [`BoxConstraint`](@ref) | + any penalty |
| Parallel imaging without sparsity assumptions | [`L2Image`](@ref) | — |
| Multi-coil data with no sensitivity maps and no calibration region | [`StructuredLowRank`](@ref) | + [`TotalVariation2D`](@ref) on the combined image |

## References

Sparsity and total variation:
- Lustig, M., Donoho, D., & Pauly, J. M. (2007). *Sparse MRI: The application of compressed sensing for rapid MR imaging.* Magnetic Resonance in Medicine, 58(6), 1182-1195. — the original CS-MRI formulation with ℓ₁-wavelet and total variation.
- Block, K. T., Uecker, M., & Frahm, J. (2007). *Undersampled radial MRI with multiple coils: Iterative image reconstruction using a total variation constraint.* Magnetic Resonance in Medicine, 57(6), 1086-1098.
- Fessler, J. A. (2010). *Model-based image reconstruction for MRI.* IEEE Signal Processing Magazine, 27(4), 81-89. — quadratic and edge-preserving penalties, non-negativity.
- Charbonnier, P., Blanc-Féraud, L., Aubert, G., & Barlaud, M. (1997). *Deterministic edge-preserving regularization in computed imaging.* IEEE Transactions on Image Processing, 6(2), 298-311. — the Huber-type potential behind [`EdgePreservingRoughness2D`](@ref).
- Chambolle, A., & Lions, P.-L. (1997). *Image recovery via total variation minimization and related problems.* Numerische Mathematik, 76(2), 167-188. — infimal convolution of first- and second-order TV.
- Bredies, K., Kunisch, K., & Pock, T. (2010). *Total generalized variation.* SIAM Journal on Imaging Sciences, 3(3), 492-526. — [`TotalGeneralizedVariation2D`](@ref).
- Knoll, F., Bredies, K., Pock, T., & Stollberger, R. (2011). *Second order total generalized variation (TGV) for MRI.* Magnetic Resonance in Medicine, 65(2), 480-491.
- Blumensath, T., & Davies, M. E. (2009). *Iterative hard thresholding for compressed sensing.* Applied and Computational Harmonic Analysis, 27(3), 265-274. — [`L0Image`](@ref), [`L0Wavelet2D`](@ref) and [`L0Wavelet3D`](@ref).

Dynamic imaging:
- Lustig, M., Santos, J. M., Donoho, D. L., & Pauly, J. M. (2006). *k-t SPARSE: High frame rate dynamic MRI exploiting spatio-temporal sparsity.* Proc. ISMRM. — sparsity in the temporal Fourier domain ([`L1TemporalFourier`](@ref)).
- Feng, L., Grimm, R., Block, K. T., et al. (2014). *Golden-angle radial sparse parallel MRI: Combination of compressed sensing, parallel imaging, and golden-angle radial sampling for fast and flexible dynamic volumetric MRI.* Magnetic Resonance in Medicine, 72(3), 707-717. — temporal total variation ([`TemporalTotalVariation`](@ref)).
- Otazo, R., Candès, E., & Sodickson, D. K. (2015). *Low-rank plus sparse matrix decomposition for accelerated dynamic MRI with separation of background and dynamic components.* Magnetic Resonance in Medicine, 73(3), 1125-1136. — the L+S model, see [Image Decomposition](image_decomposition.md).

Low-rank models:
- Liang, Z.-P. (2007). *Spatiotemporal imaging with partially separable functions.* Proc. IEEE ISBI, 988-991. <https://doi.org/10.1109/ISBI.2007.357020> — the partially separable / globally low-rank model behind [`LowRank`](@ref) and [`RankLimit`](@ref).
- Trzasko, J. D., & Manduca, A. (2011). *Local versus global low-rank promotion in dynamic MRI series reconstruction.* Proc. ISMRM, 4371. — [`LocallyLowRank`](@ref).
- Zhang, T., Pauly, J. M., & Levesque, I. R. (2015). *Accelerating parameter mapping with a locally low rank constraint.* Magnetic Resonance in Medicine, 73(2), 655-661.
- Ong, F., & Lustig, M. (2016). *Beyond low rank + sparse: Multiscale low rank matrix decomposition.* IEEE Journal of Selected Topics in Signal Processing, 10(4), 672-687. — [`MultiScaleLowRank`](@ref).
- Bauschke, H. H., Goebel, R., Lucet, Y., & Wang, X. (2008). *The proximal average: Basic theory.* SIAM Journal on Optimization, 19(2), 766-785. — the construction [`MultiScaleLowRank`](@ref) uses to combine the scales.
- Shin, P. J., Larson, P. E. Z., Ohliger, M. A., et al. (2014). *Calibrationless parallel imaging reconstruction based on structured low-rank matrix completion.* Magnetic Resonance in Medicine, 72(4), 959-970. <https://doi.org/10.1002/mrm.24997> — SAKE, the `max_rank` form of [`StructuredLowRank`](@ref).
- Haldar, J. P. (2014). *Low-rank modeling of local k-space neighborhoods (LORAKS) for constrained MRI.* IEEE Transactions on Medical Imaging, 33(3), 668-681. <https://doi.org/10.1109/TMI.2013.2293974> — LORAKS, whose C-matrix penalty is the `λ` form of [`StructuredLowRank`](@ref) and whose S- and G-matrices are its `structure = :s` and `:g`.
- Haldar, J. P., & Zhuo, J. (2016). *P-LORAKS: Low-rank modeling of local k-space neighborhoods with parallel imaging data.* Magnetic Resonance in Medicine, 75(4), 1499-1514. — the multi-channel form of the C and S matrices.
- Jin, K. H., Lee, D., & Ye, J. C. (2016). *A general framework for compressed sensing and parallel MRI using annihilating filter based low-rank Hankel matrix.* IEEE Transactions on Computational Imaging, 2(4), 480-495. — ALOHA (the `weights` argument of [`StructuredLowRank`](@ref)).

Joint sparsity and prior images:
- Majumdar, A., & Ward, R. K. (2011). *Joint reconstruction of multiecho MR images using correlated sparsity.* Magnetic Resonance Imaging, 29(7), 899-906. — [`JointSparsity`](@ref).
- Huang, J., Chen, C., & Axel, L. (2014). *Fast multi-contrast MRI reconstruction.* Magnetic Resonance Imaging, 32(10), 1344-1352.
- Chen, G.-H., Tang, J., & Leng, S. (2008). *Prior image constrained compressed sensing (PICCS).* Medical Physics, 35(2), 660-663. — [`ReferencePrior`](@ref).

Learned and denoiser-based priors:
- Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). *Plug-and-play priors for model based reconstruction.* Proc. IEEE GlobalSIP, 945-948. <https://doi.org/10.1109/GlobalSIP.2013.6737048> — [`PlugAndPlay`](@ref).
- Ahmad, R., Bouman, C. A., Buzzard, G. T., et al. (2020). *Plug-and-play methods for magnetic resonance imaging.* IEEE Signal Processing Magazine, 37(1), 105-116.

Algorithms:
- Beck, A., & Teboulle, M. (2009). *A fast iterative shrinkage-thresholding algorithm for linear inverse problems.* SIAM Journal on Imaging Sciences, 2(1), 183-202. — FISTA.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). *Distributed optimization and statistical learning via the alternating direction method of multipliers.* Foundations and Trends in Machine Learning, 3(1), 1-122. — ADMM.

## Regularizers Not Currently Available

The following terms appear in the literature and in other reconstruction packages but are not implemented here, because they need building blocks the package does not yet have:

- **Learned reconstruction networks** (unrolled networks, end-to-end variational networks): these replace the reconstruction, not the regularizer. A trained *denoiser* can be used today through [`PlugAndPlay`](@ref).

Note that [`PlugAndPlay`](@ref) supplies the mechanism but no denoisers: any callable
`denoiser(image, σ)` — BM3D, a neural network, anything — can be plugged in, but none is bundled.

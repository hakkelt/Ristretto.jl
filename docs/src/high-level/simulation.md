# Simulation Tools

*Tutorial: [Simulation](../tutorials/03_simulation.md).*

Ristretto provides comprehensive tools for simulating MRI acquisitions. These are essential for testing reconstruction algorithms, teaching MRI concepts, and prototyping new acquisition strategies.

## Why Simulate?

Simulation is useful for:

- **Testing reconstruction algorithms** without needing real MRI data
- **Understanding MRI physics** through hands-on experimentation
- **Prototyping new acquisition strategies** before scanner implementation
- **Teaching and learning** MRI concepts interactively
- **Benchmarking** reconstruction methods with known ground truth

## Overview: Complete Simulation Pipeline

```@setup imports
using Ristretto
using GeometricMedicalPhantoms
using MIRTjim: jim
using Plots
using Random

Random.seed!(0)
```

A typical simulation workflow:

```@example imports
using Ristretto

# 1. Create a phantom (ground truth image), area-sampled on a grid about 1.6× finer than the
#    256 × 256 reconstruction (see "Avoiding the inverse crime" below)
img = create_shepp_logan_phantom(404, 404, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32, supersample = 4)

# 2. Generate coil sensitivity maps on the phantom's grid
smaps = coil_sensitivities(404, 404, 8)

# 3. Create a subsampling pattern on the reconstruction grid
pdf = VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.1)
pattern = create_sampling_pattern(pdf, (256, 256))

# 4. Simulate the acquisition
acq = AcquisitionInfo(is3D=false,
                      image_size=(256, 256),
                      sensitivity_maps=smaps,
                      subsampling=pattern)
acq_with_data = simulate_acquisition(img, acq; keep_sensitivity_maps = true)

# 5. Reconstruct and compare
img_recon = reconstruct(acq_with_data, IterativeReconstruction(L1Wavelet2D(5e-3)); verbosity = Silent())
nothing # hide
```

The returned acquisition carries the maps resampled to the reconstruction grid only because
`keep_sensitivity_maps = true` asks for them; with real data they would be estimated
(`estimate_sensitivities`), and so would they in a simulation that should not reuse the model it
was generated with.

## Phantoms

Phantoms are synthetic images that serve as ground truth for testing.

### Shepp-Logan Phantom

The classic test phantom for MRI reconstruction. Ristretto uses phantoms from the [GeometricMedicalPhantoms.jl](https://github.com/hakkelt/GeometricMedicalPhantoms.jl) package.

```@example imports
using MIRTjim: jim

# 2D Shepp-Logan
img = create_shepp_logan_phantom(256, 256, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)

# 3D Shepp-Logan
img_3d = create_shepp_logan_phantom(128, 128, 64; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
jim(img_3d; title="Shepp-Logan Phantom", nrow=4, size=(1200,300))
savefig("3D_shepp_logan_phantom.png"); nothing # hide
```

![3D_shepp_logan_phantom.png](3D_shepp_logan_phantom.png)

## Coil Sensitivity Maps

Sensitivity maps model the spatial response of receiver coils in parallel imaging.

```@docs
coil_sensitivities
```

```@example imports
# Generate sensitivity maps for 8 coils
smaps = coil_sensitivities(256, 256, 8)

# 3D sensitivity maps
smaps_3d = coil_sensitivities(128, 128, 64, 8)

# Visualize 2D coil sensitivities
jim(smaps; title="Coil Sensitivity Maps", nrow=1, size=(1400, 200))
savefig("coil_sensitivity_maps.png"); nothing # hide
```

![coil_sensitivity_maps.png](coil_sensitivity_maps.png)

**What you get:**
- Smooth, realistic sensitivity patterns
- Each coil has higher sensitivity near its location
- Proper phase variations
- Returns ComplexF32 array with shape (nx, ny, [nz,] ncoils)

## Subsampling Patterns

Subsampling patterns determine which k-space locations are measured. Each strategy below is a
`Subsampling` object; `create_sampling_pattern` draws one realisation of it.

```@docs
create_sampling_pattern
to_displayable_mask
```

### Uniform Cartesian Sampling

```@docs
UniformRandomSampling
```

#### Example: 2D Uniform Random Sampling
```@example imports
using Plots

# Center fraction = 0.1 (10% fully sampled center, default)
pdf = UniformRandomSampling(3.0)
pattern = create_sampling_pattern(pdf, (256, 256))
p1 = jim(to_displayable_mask(pattern, (256, 256)); title = "cf=0.1")

# Center fraction = 0.3 (30% fully sampled center)
pdf = UniformRandomSampling(3.0, 0.3)
pattern = create_sampling_pattern(pdf, (256, 256))
p2 = jim(to_displayable_mask(pattern, (256, 256)); title = "cf=0.3")

# 2D subsampling: freq encoding also subsampled (not realistic, for demo)
pdf = UniformRandomSampling(3.0)
pattern = create_sampling_pattern(pdf, (256, 256), subsample_freq_encoding=true)
p3 = jim(to_displayable_mask(pattern, (256, 256)); title = "Freq encoding subsampled")

jim(p1, p2, p3; layout = (1, 3), size = (1000, 300))
savefig("2D_uniform_random_sampling_weights.png"); nothing # hide
```

![2D_uniform_random_sampling_weights.png](2D_uniform_random_sampling_weights.png)

#### Example: 3D Uniform Random Sampling
```@example imports
pdf = UniformRandomSampling(4.0, 0.1)
pattern_3d = create_sampling_pattern(pdf, (128, 128, 64))
mask = zeros(Bool, 128, 128, 64)
mask[pattern_3d...] .= true
p1 = jim(mask[:, :, 32]; title="x-y plane")
p2 = jim(mask[:, 64, :]; title="x-z plane")
p3 = jim(mask[64, :, :]; title="y-z plane")
jim(p1, p2, p3; layout=(1,3), size=(1000,300))
savefig("3D_uniform_random_sampling_weights.png"); nothing # hide
```

![3D_uniform_random_sampling_weights.png](3D_uniform_random_sampling_weights.png)

### Variable Density Sampling

The most common pattern for compressed sensing. There are many options for generating variable density patterns.

```@docs
GaussianDistribution
PolynomialDistribution
VariableDensitySampling
```

#### Example: Different Variable Density Patterns in 2D

```@example imports
gaussian_pdf₁ = VariableDensitySampling(GaussianDistribution(1/3), 3.0)
gaussian_pattern₁ = create_sampling_pattern(gaussian_pdf₁, (256, 256))
W = Ristretto.construct_weights(gaussian_pdf₁, (256,))

p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(gaussian_pattern₁, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Gaussian std=1/3", size = (700, 300))
savefig("gaussian_sampling_pattern_1.png"); nothing # hide
```

![gaussian_sampling_pattern_1.png](gaussian_sampling_pattern_1.png)

```@example imports
gaussian_pdf₂ = VariableDensitySampling(GaussianDistribution(1/5), 3.0)
gaussian_pattern₂ = create_sampling_pattern(gaussian_pdf₂, (256, 256))
W = Ristretto.construct_weights(gaussian_pdf₂, (256,))

p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(gaussian_pattern₂, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Gaussian std=1/5", size = (700, 300))
savefig("gaussian_sampling_pattern_2.png"); nothing # hide
```

![gaussian_sampling_pattern_2.png](gaussian_sampling_pattern_2.png)

```@example imports
poly_pdf₁ = VariableDensitySampling(PolynomialDistribution(2), 3.0)
poly_pattern₁ = create_sampling_pattern(poly_pdf₁, (256, 256))

W = Ristretto.construct_weights(poly_pdf₁, (256,))
p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(poly_pattern₁, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Polynomial p=2", size = (700, 300))
savefig("polynomial_sampling_pattern_1.png"); nothing # hide
```

![polynomial_sampling_pattern_1.png](polynomial_sampling_pattern_1.png)

```@example imports
poly_pdf₂ = VariableDensitySampling(PolynomialDistribution(4), 3.0)
poly_pattern₂ = create_sampling_pattern(poly_pdf₂, (256, 256))
W = Ristretto.construct_weights(poly_pdf₂, (256,))
p1 = plot(W; legend = false)
p2 = jim(to_displayable_mask(poly_pattern₂, (256, 256)))
jim(p1, p2; layout=(1,2), plot_title="Polynomial p=4", size = (700, 300))
savefig("polynomial_sampling_pattern_2.png"); nothing # hide
```

![polynomial_sampling_pattern_2.png](polynomial_sampling_pattern_2.png)

#### Example: Variable Density in 3D

```@example imports
poly_pdf = VariableDensitySampling(PolynomialDistribution(2), 3.0)
poly_pattern = create_sampling_pattern(poly_pdf, (256, 256, 256))
mask = zeros(Bool, 256, 256, 256)
mask[poly_pattern...] .= true
p1 = jim(mask[:, :, 128]; title="x-y plane")
p2 = jim(mask[:, 128, :]; title="x-z plane")
p3 = jim(mask[128, :, :]; title="y-z plane")
jim(p1, p2, p3; layout=(1,3), size=(900,200))
savefig("3D_variable_density_sampling_weights.png"); nothing # hide
```

![3D_variable_density_sampling_weights.png](3D_variable_density_sampling_weights.png)

### Poisson Disk Sampling

Spatially uniform but avoiding clustering.

```@docs
PoissonDiskSampling
```

```@example imports
pdf = PoissonDiskSampling(3.0)
pattern = create_sampling_pattern(pdf, (256, 256), subsample_freq_encoding=true)
jim(to_displayable_mask(pattern, (256, 256)); title="Poisson Disk Sampling", size = (300, 300))
savefig("poisson_disk_sampling_pattern.png"); nothing # hide
```

![poisson_disk_sampling_pattern.png](poisson_disk_sampling_pattern.png)

**Properties:**
- Maintains minimum distance between samples
- More uniform coverage than random
- Good incoherence properties

### Regular Lattice Sampling

The product parallel-imaging pattern: every R-th phase encode, optionally with a fully sampled
autocalibration (ACS) band. Unlike the random generators it is deterministic, and it is the
pattern [`GRAPPA`](@ref) requires — its kernel is defined by a fixed geometric relation between a
hole and its neighbours, which only a regular lattice has.

```@docs
RegularLatticeSampling
```

### Partial Fourier Sampling

A contiguous band from one end of k-space, everything past it left unacquired. It exploits the
Hermitian symmetry of k-space instead of coil encoding, so it is its own scheme rather than a
modifier of the lattice; reconstruct it with [`Homodyne`](@ref) or [`POCS`](@ref).

```@docs
PartialFourierSampling
```

```@example imports
# R = 3 with a 10% ACS band: the GRAPPA/SPIRiT calibration pattern
grappa_pdf = RegularLatticeSampling(3; center_fraction = 0.1)
grappa_pattern = create_sampling_pattern(grappa_pdf, (256, 256))

# 6/8 partial Fourier
pf_pdf = PartialFourierSampling(0.75)
pf_pattern = create_sampling_pattern(pf_pdf, (256, 256))

jim(
    jim(to_displayable_mask(grappa_pattern, (256, 256)); title = "R=3 + ACS"),
    jim(to_displayable_mask(pf_pattern, (256, 256)); title = "75% partial Fourier");
    layout = (1, 2), size = (600, 300)
)
savefig("systematic_sampling_pattern.png"); nothing # hide
```

![systematic_sampling_pattern.png](systematic_sampling_pattern.png)

Over two subsampled dimensions (a 3D acquisition) the lattice acceleration is factored into a
stride per dimension, as close to equal as its divisors allow: `4` becomes 2×2, `3` becomes 3×1.

## Non-Cartesian Trajectories

Ready-made generators for the common non-Cartesian sampling patterns, returned as
`NamedDimsArray`s in exactly the layout `AcquisitionInfo`/[`simulate_acquisition`](@ref) expect
(coordinate axis first, normalized to `[-0.5, 0.5)`, NFFT.jl convention):

```@docs
radial_trajectory
stack_of_stars_trajectory
kooshball_trajectory
phyllotaxis_trajectory
spiral_trajectory
floret_trajectory
sparkling_trajectory
```

`radial_trajectory`, `stack_of_stars_trajectory`, `kooshball_trajectory` and `phyllotaxis_trajectory` take
`center_out = true` for half spokes (center-out radial, as in UTE), which start on `k = 0`.

The spoke ordering and the spiral growth law are chosen with a type rather than a symbol, so each
carries its own parameters and a typo is a `MethodError` instead of a runtime `ArgumentError`:

```@docs
LinearOrdering
GoldenAngle
TinyGoldenAngle
Archimedean
VariableDensity
```

```@example imports
using NamedDims

traj = radial_trajectory(128, 96; ordering = GoldenAngle())
acq_radial = AcquisitionInfo(; trajectory = traj, image_size = (256, 256))
data_radial = simulate_acquisition(img, acq_radial; inverse_crime_check = false)
size(data_radial.kspace_data)
```

## Simulate Acquisition

```@docs
simulate_acquisition
```

### Avoiding the inverse crime

Simulating data with the operator that later reconstructs them, on the reconstruction's own grid,
is the *inverse crime* [1]: the data fit the model exactly, so a reconstruction looks better than
it would on measured data. `simulate_acquisition` therefore treats `acq_info.image_size` as the
reconstruction grid and the phantom's own size as the grid the data are simulated on. From a
finer phantom it keeps only the frequencies the reconstruction grid can represent (Cartesian), or
samples the trajectory at the same physical frequencies on the phantom's grid (non-Cartesian).
A phantom of the reconstruction's size gets a warning, and so does an integer multiple of it,
each checked per spatial axis; `inverse_crime_check = false` silences both where the consistency
is what is being tested.

How fine is fine enough was measured on the modified Shepp–Logan phantom, whose continuous Fourier
transform is known in closed form, so data can be simulated without any grid
(`benchmark/inverse_crime/study.jl`). Reconstruction grid 128 × 128; the table gives the relative
error of data simulated from rasterized phantoms of `s` times that size. *Point*: each voxel takes
the value at its centre. *Area*: the mean of 4 × 4 sub-voxel samples (GeometricMedicalPhantoms'
`supersample = 4`).

| ratio `s` | phantom | Cartesian, point | Cartesian, area | radial, point | radial, area |
|---|---|---|---|---|---|
| 1 (inverse crime) | 128² | 21.2 % | 8.2 % | 3.0 % | 1.1 % |
| 1.3 | 166² | 14.2 % | 5.1 % | 2.0 % | 0.6 % |
| 1.5 | 192² | 11.2 % | 3.5 % | 1.7 % | 0.5 % |
| 1.58 | 202² | 10.8 % | 3.2 % | 1.6 % | 0.4 % |
| 1.7 | 218² | 8.7 % | 2.6 % | 1.2 % | 0.3 % |
| 2 | 256² | 7.7 % | 2.0 % | 1.2 % | 0.3 % |
| 3 | 384² | 4.2 % | 0.9 % | 0.7 % | 0.1 % |

Area sampling matters more than the ratio: it cuts the error three- to fourfold. Larger is always
more accurate, at a cost that grows as `s^D`, so the ratio is a cost/accuracy choice. The
recommendation is the smallest area-sampled ratio whose error reaches the noise level of 30 dB
data (3.2 %): **about 1.6 times the reconstruction size per axis, area-sampled**, e.g. a 202²
phantom for a 128² reconstruction. Round ratios are best avoided: at `s = 2` every voxel centre
of the reconstruction grid is also one of the phantom, which makes errors measured against a
point-sampled phantom about 7 % optimistic.

The effect on a reconstruction was measured as in [2]: the signal-to-error ratio of a total
variation reconstruction from 3-fold undersampled data simulated from the rasterized phantom,
minus that from the analytic data, both against the area-sampled 128² phantom. Area-sampled
phantoms come out 1.3–1.5 dB optimistic for `1.5 ≤ s ≤ 2` (0.6 dB at 1.3 and at 3), point-sampled
ones 3–5 dB pessimistic; the inverse crime itself is 6.8 dB optimistic with an area-sampled
phantom. Part of the remaining bias is the box filter that area sampling applies, which brings the
data closer to the area-sampled reference than the analytic data are.

Coil sensitivity maps act on the phantom, so they are made at its size. The returned acquisition
carries none unless `keep_sensitivity_maps = true` (it then carries them resampled to the
reconstruction grid): reusing the maps that simulated the data is part of the same crime, and
with measured data they are estimated (`estimate_sensitivities`). Dropping them is logged as an
info message, since a reconstruction without maps treats the coil axis as a batch axis;
`keep_sensitivity_maps = false` drops them silently.

A reconstruction from such data is scored against the same object rasterized area-sampled on the
reconstruction grid. Neither the fine phantom (a different size) nor a point-sampled phantom of
the reconstruction's size will do: against the latter every edge counts as error, which is the
3–5 dB pessimism above.

```@example imports
img_fine = create_shepp_logan_phantom(202, 202, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32, supersample = 4)
acq_128 = AcquisitionInfo(is3D = false, image_size = (128, 128), sensitivity_maps = coil_sensitivities(202, 202, 8))
data_128 = simulate_acquisition(img_fine, acq_128)
truth_128 = create_shepp_logan_phantom(128, 128, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32, supersample = 4)
size(data_128.kspace_data), data_128.sensitivity_maps, size(truth_128)
```

## Advanced Simulation

### Adding Noise

`add_noise` adds complex Gaussian noise to k-space data, either as a target SNR (in dB, relative
to the RMS of the data) or as an absolute standard deviation. It accepts a plain array, a
`NamedDimsArray`, or an `AcquisitionInfo` directly (in which case a *copy* with noisy
`kspace_data` is returned, via the same copy-constructor pattern as `AcquisitionInfo(acq; ...)`):

```@docs
add_noise
```

Noise can also be added to an **image** rather than to k-space, targeting the clinical SNR — a bare
ratio measured the way a scanner acceptance test measures it, rather than a decibel figure relative
to the data RMS. [`estimate_snr`](@ref) measures that same quantity back from an image:

```@docs
estimate_snr
```

```@example imports
noisy_image = add_noise(img; snr = 20)
estimate_snr(noisy_image)
```

Both regions are boxes whose size the caller gives in voxels — a centred one for the signal, one in
each corner for the noise — so nothing about where they sit depends on the image being measured.
They can be displayed rather than trusted:

```@docs
snr_masks
```

```@example imports
# Target SNR in dB
noisy_acq = add_noise(acq_with_data; snr_db = 20)

# Or an absolute noise standard deviation
noisy_acq2 = add_noise(acq_with_data; noise_std = 0.02)

# Reconstruct noisy data
img_recon = reconstruct(noisy_acq, IterativeReconstruction(L1Wavelet2D(5e-3)); verbosity = Silent())
nothing # hide
```

### Subsampling: Indexing Expressions vs. Boolean Masks

`subsampling` accepts two forms (see [AcquisitionInfo](@ref) for the full set, including plain
boolean masks):

1. **A tuple of indexing expressions** (`Colon`, ranges, or integer vectors) — the **default**
   idiom for anything with regular structure. It reads directly as "keep these indices along
   each dimension", is cheaper to construct and store than a full mask, and composes naturally
   with `Base`'s indexing:

   ```@example imports
   ny = 256

   # Partial Fourier: keep the first 65% of phase-encoding lines
   pf_pattern = (:, 1:round(Int, 0.65 * ny))

   # Uniform GRAPPA-style undersampling (every 4th line) with a fully sampled ACS block —
   # a range/step expression reads directly as the acceleration factor and ACS width, where a
   # mask only shows the result of them.
   R, acs_half_width = 4, 12
   center = div(ny, 2)
   grappa_lines = sort(union(1:R:ny, (center - acs_half_width):(center + acs_half_width)))
   grappa_pattern = (:, grappa_lines)
   ```

2. **A boolean mask** (or `(:, mask)` for a fully sampled frequency-encoding dimension) — the
   special case for genuinely irregular or random patterns, where there is no indexing
   expression to write down. `create_sampling_pattern` (above) returns this form.

Both forms are passed the same way:

```@example imports
acq_pf = AcquisitionInfo(nothing; is3D=false, image_size=(ny, ny), subsampling=pf_pattern)
acq_grappa = AcquisitionInfo(nothing; is3D=false, image_size=(ny, ny), subsampling=grappa_pattern)
nothing # hide
```

## References

1. Kaipio, J., Somersalo, E., "Statistical inverse problems: discretization, model reduction and
   inverse crimes", Journal of Computational and Applied Mathematics 198(2):493–504 (2007).
   <https://doi.org/10.1016/j.cam.2005.09.027>
2. Guerquin-Kern, M., Lejeune, L., Pruessmann, K. P., Unser, M., "Realistic analytical phantoms
   for parallel magnetic resonance imaging", IEEE Transactions on Medical Imaging 31(3):626–636
   (2012). <https://doi.org/10.1109/TMI.2011.2174158>

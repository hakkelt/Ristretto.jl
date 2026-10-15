# Simulation Tools

The reference for the simulation tools: phantoms, coil maps, sampling patterns, trajectories,
`simulate_acquisition` and noise. [Simulation](../tutorials/03_simulation.md) shows each of them
in pictures, and [Non-Cartesian acquisitions](../tutorials/08_non_cartesian.md) the trajectories.

## Phantoms

Phantoms come from [GeometricMedicalPhantoms.jl](https://github.com/hakkelt/GeometricMedicalPhantoms.jl):
`create_shepp_logan_phantom(nx, ny, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)`
in 2D, `create_shepp_logan_phantom(nx, ny, nz; ...)` in 3D, and `supersample = 4` for the
area-sampled phantom that simulation needs (see [Avoiding the inverse crime](#Avoiding-the-inverse-crime)).

## Coil Sensitivity Maps

Sensitivity maps model the spatial response of receiver coils in parallel imaging.

```@docs
coil_sensitivities
```

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

### Variable Density Sampling

The most common pattern for compressed sensing. There are many options for generating variable density patterns.

```@docs
GaussianDistribution
PolynomialDistribution
VariableDensitySampling
```

### Poisson Disk Sampling

Spatially uniform but avoiding clustering.

```@docs
PoissonDiskSampling
```

### Regular Lattice Sampling

The product parallel-imaging pattern: every R-th phase encode, optionally with a fully sampled
autocalibration (ACS) band. Unlike the random generators it is deterministic, and it is the
pattern [`GRAPPA`](@ref) requires — its kernel is defined by a fixed geometric relation between a
hole and its neighbours, which only a regular lattice has.

```@docs
RegularLatticeSampling
```

Over two subsampled dimensions (a 3D acquisition) the lattice acceleration is factored into a
stride per dimension, as close to equal as its divisors allow: `4` becomes 2×2, `3` becomes 3×1.

### Partial Fourier Sampling

A contiguous band from one end of k-space, everything past it left unacquired. It exploits the
Hermitian symmetry of k-space instead of coil encoding, so it is its own scheme rather than a
modifier of the lattice; reconstruct it with [`Homodyne`](@ref) or [`POCS`](@ref).

```@docs
PartialFourierSampling
```

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

```julia
traj = radial_trajectory(128, 96; ordering = GoldenAngle())
acq_radial = AcquisitionInfo(; trajectory = traj, image_size = (256, 256))
data_radial = simulate_acquisition(phantom, acq_radial)
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

```julia
phantom(n) = create_shepp_logan_phantom(n, n, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32, supersample = 4)
acq = AcquisitionInfo(is3D = false, image_size = (128, 128), sensitivity_maps = coil_sensitivities(202, 202, 8))
data = simulate_acquisition(phantom(202), acq)   # 128² k-space, no sensitivity maps
truth = phantom(128)                             # what a reconstruction of `data` is scored against
```

## Noise

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

Both regions are boxes whose size the caller gives in voxels — a centred one for the signal, one in
each corner for the noise — so nothing about where they sit depends on the image being measured.
They can be displayed rather than trusted:

```@docs
snr_masks
```

```julia
noisy_acq = add_noise(acq; snr_db = 20)       # k-space noise, relative to the data RMS
noisy_acq = add_noise(acq; noise_std = 0.02)  # or an absolute standard deviation
noisy_image = add_noise(image; snr = 20)      # image noise at a clinical SNR
estimate_snr(noisy_image)
```

## Hand-Written Patterns

`subsampling` also takes a tuple of indexing expressions (`Colon`, ranges, integer vectors), the
natural form for any pattern with regular structure, such as every 4th line plus an ACS block:
`(:, sort(union(1:4:ny, (ny ÷ 2 - 12):(ny ÷ 2 + 12))))`. A boolean mask, or `(:, mask)`, is the
form for irregular patterns and the one `create_sampling_pattern` returns.
[AcquisitionInfo](@ref) lists every accepted form; the
[Simulation tutorial](../tutorials/03_simulation.md) builds them by hand.

## References

1. Kaipio, J., Somersalo, E., "Statistical inverse problems: discretization, model reduction and
   inverse crimes", Journal of Computational and Applied Mathematics 198(2):493–504 (2007).
   <https://doi.org/10.1016/j.cam.2005.09.027>
2. Guerquin-Kern, M., Lejeune, L., Pruessmann, K. P., Unser, M., "Realistic analytical phantoms
   for parallel magnetic resonance imaging", IEEE Transactions on Medical Imaging 31(3):626–636
   (2012). <https://doi.org/10.1109/TMI.2011.2174158>

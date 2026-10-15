# [Image Reconstruction](@id reconstruction)

`reconstruct(acq, method)` turns an [`AcquisitionInfo`](@ref) into an image. `method` defaults to
[`DirectReconstruction`](@ref), the adjoint ``\mathcal{A}^H y``; an [`IterativeReconstruction`](@ref)
solves ``\min_x \tfrac12\|\mathcal{A}x - y\|_2^2 + \sum_i \lambda_i R_i(x)`` with the
[regularizers](regularization.md) it is given and an [algorithm](algorithms.md) it picks or is
told. This page covers the call, its result and the run settings. Batch dimensions are split into
independent solves automatically ([Task Splitting](task_splitting.md)).
[Getting started](../tutorials/01_getting_started.md) and
[Algorithms and configuration](../tutorials/06_algorithms_and_configuration.md) show them in use.

## API Reference

```@docs
reconstruct
DirectReconstruction
IterativeReconstruction
ReconstructionConfig
```

## The result: `ReconImage`

```@setup recon
using Ristretto
using GeometricMedicalPhantoms: create_shepp_logan_phantom, MRISheppLoganIntensities
using Random
Random.seed!(0)
n, nc, nt = 64, 8, 8
phantom = create_shepp_logan_phantom(n, n, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32)
smaps = coil_sensitivities(n, n, nc)
pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 3.0, 0.08), (n, n))
simulate(x, acq) = add_noise(simulate_acquisition(x, acq; inverse_crime_check = false, keep_sensitivity_maps = true); snr_db = 30)
acq = simulate(phantom, AcquisitionInfo(; image_size = (n, n), subsampling = pattern, sensitivity_maps = smaps))
using NamedDims
ksp_ms = NamedDimsArray{(:kx, :ky, :coil, :z)}(rand(ComplexF32, n, n, nc, 4))
maps_ms = NamedDimsArray{(:x, :y, :coil, :z)}(rand(ComplexF32, n, n, nc, 4))
acq_ms = AcquisitionInfo(ksp_ms; sensitivity_maps = maps_ms, header = (; fov = (220, 220), slice_thickness = 5))
```

`reconstruct` returns a [`ReconImage`](@ref): the image (a `NamedDimsArray` inside when the
k-space was named) with a copy of the acquisition's [`Header`](@ref). It behaves as the array it
holds, and keyword indexing returns a `ReconImage` whose geometry describes the part selected:

```@example recon
acq_cartesian = acq # hide
acq = acq_ms # hide
img = reconstruct(acq)                  # acq: named (:kx, :ky, :z) k-space with a header
Ristretto.header(img[z = 2]).offset    # moved by one slice spacing
Ristretto.header(img[x = 17:48]).fov   # the field of view of the crop
```

Tagging the image (`settag!(img, :reader, "A")`) never changes the acquisition. `Array(img)` is
a plain array, and broadcasting acts on the image data and returns a plain array.

```@docs
ReconImage
```

## Configuration Control

There are two separate places a parameter can live, and which one it belongs to is decided by
one question: *does it mean anything without knowing the method?*

- **Method parameters** — `maxit`, `reltol`, `algorithm`, and everything else only a particular
  method can act on — go to that method's constructor. They are keyword-only there.
- **Run settings** — scaling, output, threading, task splitting, FFT planning (`fft_planning`,
  see [Planning FFTs](@ref)) — go to `ReconstructionConfig`, or straight to `reconstruct` as
  keywords.

Passing `maxit`, `reltol` or `algorithm` to `reconstruct` throws rather than being silently
ignored, which is what happened before this split.

```@example recon
acq = acq_cartesian # hide
method = IterativeReconstruction(L1Wavelet2D(5e-3); maxit = 50, reltol = 1e-5)   # method parameters
reconstruct(acq, method; scaling = QuantileScaling(), verbosity = Silent())      # run settings as keywords
config = ReconstructionConfig(; scaling = QuantileScaling())                      # or reusable
reconstruct(acq, method; config, verbosity = Verbose())                          # keywords override it
nothing # hide
```

### Important Configuration Options

#### Iteration Control

`maxit`, `reltol` and `algorithm` are method parameters; [Optimization Algorithms](algorithms.md#Iterations-and-Tolerance)
covers what they mean and how they interact with the algorithm's own settings.

#### Output

`verbosity` takes one of three modes: `Silent()` (the default), `ProgressBar()`, or
`Verbose(; freq, printfunc)`, a log line every `freq` iterations written through `printfunc`
(`println` by default; pass a function that collects the lines to capture them). It also accepts
`true`/`false` and the symbols `:verbose`, `:progress`, `:silent`.

```@docs
Verbosity
Silent
ProgressBar
Verbose
```

#### Scaling

The k-space data are divided by a scale `s` before the solve and the image is multiplied by it
afterwards, so a regularization weight `λ` acts on data of a fixed intensity. For a positively
homogeneous penalty (every norm-based regularizer) dividing the data by `s` is the same problem as
multiplying `λ` by `s`, so each rule below is equally a rule for the effective `λ`. Every rule but
`SystemMatrixBasedScaling`, `FixedScaling` and `NoScaling` is proportional to the data, which is
what makes the result independent of the data's intensity: `reconstruct` of `c·y` is `c` times
`reconstruct` of `y`.

| scaling | `s` | source | robust to a few hot voxels |
|---|---|---|---|
| [`QuantileScaling`](@ref) (default) | 99th percentile of `\|𝒜ᴴy\|` | percentile normalization | yes, to about 1 % of the voxels |
| [`BartScaling`](@ref) | 90th percentile of `\|𝒜ᴴy\|`, or its maximum when the spread is large | BART `pics` | no when it selects the maximum |
| [`MaxScaling`](@ref) | maximum of `\|𝒜ᴴy\|` | fastMRI | no |
| [`StdScaling`](@ref) | standard deviation of `𝒜ᴴy` | fastMRI baselines | partly |
| [`NoiseLevelScaling`](@ref) | noise level of `𝒜ᴴy` (MAD of the finest Haar details) | Donoho and Johnstone 1994 | yes |
| [`MeasurementBasedScaling`](@ref) | mean `\|y\|` | RegularizedLeastSquares.jl | yes |
| [`KSpaceNormScaling`](@ref) | `‖y‖₂ / 100` | BART `nlinv` | yes |
| [`SystemMatrixBasedScaling`](@ref) | `trace(𝒜ᴴ𝒜) / N` | RegularizedLeastSquares.jl | not data-dependent |
| [`FixedScaling`](@ref) | a given number | | |
| [`NoScaling`](@ref) | 1 | | |

The default was chosen on the benchmark harness, whose synthetic cases span 1- and 8-coil 2D,
radial, 3D and multi-slice Cartesian, and Cartesian and radial cine. For each penalty one `λ` was
used for all cases, and the NRMSE it reached was compared with each case's own best `λ`:

| scaling | mean excess NRMSE | worst excess NRMSE | cost on a 128³ × 8 coil volume |
|---|---|---|---|
| `QuantileScaling` | 9.9 % | 45 % | 1.9 ms |
| `StdScaling` | 9.0 % | 54 % | 2.9 ms |
| `MaxScaling` | 11.5 % | 74 % | 2.1 ms |
| `BartScaling` | 12.0 % | 95 % | 5.4 ms |
| `NoiseLevelScaling` | 21 % | 166 % | 4.3 ms |
| `SystemMatrixBasedScaling` | 35 % | 202 % | 543 ms |
| `MeasurementBasedScaling` | 490 % | 6500 % | 2.7 ms |
| `KSpaceNormScaling` | 630 % | 7200 % | 2.2 ms |

`QuantileScaling`, `StdScaling`, `MaxScaling` and `BartScaling` are close on these cases, which
share one noise model. `QuantileScaling` is the default because it keeps that accuracy while a
single hot voxel moves `MaxScaling`, and `BartScaling` whenever it selects the maximum, by the
voxel's own factor. The rules that read only the k-space or only the operator do not see the gain
between k-space and image, which differs between a Cartesian and a radial encoding by six orders of
magnitude (`‖𝒜‖² ≈ 1` against `≈ 2·10⁶`).

```@docs
QuantileScaling
BartScaling
MaxScaling
StdScaling
NoiseLevelScaling
MeasurementBasedScaling
KSpaceNormScaling
SystemMatrixBasedScaling
FixedScaling
NoScaling
```

Choose one with `reconstruct(acq, method; scaling = BartScaling())`, or in a `ReconstructionConfig`.
`disable_inverse_scale_output = true` returns the image in the scaled units instead of the data's.

## Several Terms and Components

Regularizers are listed one after another in `IterativeReconstruction`, and their sum is
minimized; [Regularization](regularization.md#Combining-Multiple-Regularizers) says which solver a
combination gets. Additive models (a low-rank background plus a sparse foreground) pass
`Component`s instead; see [Image Decomposition](image_decomposition.md).

## Initial Guess

`x₀` starts the solver from a given image, in the data's units: a direct reconstruction, or the
result of a neighbouring λ in a parameter sweep.

```@example recon
x_l1 = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(5e-3)))
x = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(3e-3)); x₀ = x_l1)
nothing # hide
```

## Advanced Method Options

### Operator Scaling

By default, the encoding operator is normalized to unit norm for stable step size selection;
`IterativeReconstruction(reg; disable_operator_normalization = true)` turns that off.

### Normal Operator Optimization

For least-squares problems the gradient of the data term can be evaluated through the normal
operator $\mathcal{A}^*\mathcal{A}$ in a single pass, whenever that product has an
implementation cheaper than applying $\mathcal{A}$ and then $\mathcal{A}^*$ — which it does for
the FFT- and NFFT-based encoding operators this package builds. Nothing has to be requested:
the model always carries the plain least-squares term, and the substitution is decided when the
problem is handed to a solver, against the full set of variables the problem turns out to have.

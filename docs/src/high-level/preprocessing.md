# Pre-processing

*Tutorials: [Real Cartesian data](../tutorials/09_real_data_cartesian.md), [Non-Cartesian](../tutorials/08_non_cartesian.md).*

`Ristretto` provides functional pre-processing transforms for multi-coil MRI data.
All pre-processing functions operate on `AcquisitionInfo` instances as pure functions `AcquisitionInfo -> AcquisitionInfo`, preserving all acquisition metadata and dimension names.

```mermaid
graph LR
    Raw[Raw AcquisitionInfo] --> Prewhiten[prewhiten]
    Prewhiten --> Compress[compress_coils]
    Compress --> Sens[estimate_sensitivities]
    Sens --> Norm[normalize_sensitivity_maps]
    Norm --> Recon[reconstruct]
```

## Noise Prewhitening

Inter-coil noise correlation degrades reconstruction SNR and compromises optimal regularizer tuning.
Noise prewhitening estimates the noise covariance matrix $\Psi \in \mathbb{C}^{N_c \times N_c}$ from noise-only calibration data and decorrelates both k-space and sensitivity maps by applying $L^{-1}$, where $\Psi = L L^*$.
The whitened noise has identity covariance, so the plain $\ell_2$ data term becomes the
maximum-likelihood fit; it is also the first step of reconstruction in SNR units (Kellman &
McVeigh 2005).

```@docs
estimate_noise_covariance
prewhiten
```

### When to use:
- Multi-channel array acquisitions with non-negligible cross-coil noise coupling.
- Always apply prior to coil compression and sensitivity estimation.

## Receiver Coil Compression

Coil compression transforms multi-coil array data with $N_c$ channels into a smaller set of $N_v$ virtual coils ($N_v \ll N_c$), drastically speeding up iterative reconstruction while retaining $>99\%$ of the signal energy.
`SVDCompression` (Buehrer et al. 2007, Huang et al. 2008) applies one compression matrix to the
whole data set; `GeometricCompression` (Zhang et al. 2013) computes one per position along the
fully sampled readout and aligns them, which keeps more signal for the same number of virtual coils.

```@docs
CoilCompression
SVDCompression
GeometricCompression
compress_coils
```

### When to use:
- High-channel arrays (e.g., 32–128 channels) where iterative reconstruction runtime scales linearly with the channel count.

## Coil Sensitivity Estimation

Parallel imaging reconstruction relies on accurate spatial sensitivity profiles $S_c(r)$. `Ristretto` provides three complementary sensitivity estimation algorithms:

```@docs
SensitivityEstimation
SelfCalibrating
AdaptiveCombine
ESPIRiT
estimate_sensitivities
```

### Batch dimensions

K-space that carries batch dimensions past the coil axis — `:z` for multi-slice, `:time` for a
cine, `:contrast` for a mapping series — gets **one set of maps per slab**, returned in the same
layout as the k-space (`(:x, :y, :coil, :z)` for the multi-slice case, which is exactly the
per-slice map layout [`AcquisitionInfo`](@ref) accepts). Coil sensitivities differ from slice to
slice, so estimating them jointly would be wrong; they are estimated independently and stacked.

### Non-Cartesian acquisitions

`estimate_sensitivities(acq::NonCartesianAcquisitionInfo; ...)` calibrates radial, spiral and
arbitrary-trajectory data **directly** — no hand-rolled gridding round trip. It grids the samples
with a density-compensated NFFT adjoint (one image per coil), transforms those back onto a
Cartesian grid of `acq.image_size`, and runs the chosen estimator there, returning maps on the
centred image grid the non-Cartesian reconstruction itself uses.

- `dcf` defaults to `acq.dcf` when the acquisition carries one — vendor weights, or the output of
  [`density_compensation`](@ref) — and to `:auto` (NFFTOperators' own estimator) otherwise. It
  cannot be `nothing`: gridding without density compensation weights the calibration region by
  how densely the trajectory samples it.
- `average_dims` (default `(:time,)`) names the batch dimensions averaged over before
  calibration. The averaging happens on the samples, which the shared trajectory and the
  linearity of gridding make equivalent to averaging the gridded images, at one gridding pass
  instead of one per frame. One frame of a real-time or cine non-Cartesian series is usually
  far too undersampled to calibrate from, while the coils do not move between frames. Batch
  dimensions not named here are estimated slab by slab, as for Cartesian data; pass
  `average_dims = ()` for one set of maps per frame.

A slab whose calibration region holds no signal yields all-zero maps, and Ristretto warns rather than
returning them silently. Two file-level causes account for almost every occurrence: a header whose
`center_sample` does not match where the k-space energy is, and a 3D acquisition loaded with a
single partition, where the calibration region cannot fit along `:kz` — reconstruct that one as 2D
instead. `examples/mridata/` demonstrates both.

### Methods:
- `SelfCalibrating(; calib_size = 24)`: Smooth low-resolution calibration from central k-space auto-calibration signal (ACS) lines, normalized by root-sum-of-squares (McKenzie et al. 2002). Fastest method for Cartesian data with an ACS region.
- `AdaptiveCombine(; kernel_size = 5)`: Local array correlation matrix eigenanalysis (Walsh et al. 2000). Needs no dedicated calibration scan and provides SNR-optimal coil combination.
- `ESPIRiT(; calib_size = 24, kernel_size = 6)`: Calibration matrix null-space / subspace eigenanalysis (Uecker et al. 2014) yielding sensitivity maps with compact spatial support.

### FFT-shift convention

Sensitivity maps live in the image domain, so they must sit on the same image grid as the
reconstruction that multiplies them. Every estimator inverts centered k-space into Ristretto's *default*
convention (image origin at index 1), so the raw-array method
`estimate_sensitivities(kspace; ...)` returns maps in that convention. The `AcquisitionInfo`
method `estimate_sensitivities(acq; ...)` additionally `fftshift`s the maps onto whatever axes the
acquisition declares in `shifted_image_dims` — which raw scanner data always declares, see
[FFT-shift derivation](@ref). Prefer passing the `AcquisitionInfo`: maps estimated by hand from a
bare k-space array and attached to a shifted acquisition are rolled by half the FOV relative to
every image they multiply, which does not merely displace the reconstruction — it makes it wrong
everywhere.

## Sensitivity Map Normalization

The overall scale of a sensitivity map set is arbitrary: it depends on how the maps were estimated,
not on the anatomy. `normalize_sensitivity_maps` divides them by $\sqrt{\sum_c |S_c(r)|^2}$ so the
coil sum of squares is one wherever there is signal, which is the conventional SENSE scaling
(Pruessmann et al. 1999, Roemer et al. 1990).

```@docs
normalize_sensitivity_maps
```

```julia
acq = estimate_sensitivities(acq; method = ESPIRiT())
acq = normalize_sensitivity_maps(acq)
```

### When not to use it

The docstring above lists what normalization buys and why it is not the default. One limit it does
not state: the $\|\mathcal{A}\| \le 1$ argument holds for a plain projection-times-unitary encoding
chain, with equality for fully sampled Cartesian SENSE. Put an NUFFT, density compensation or coil
compression in the chain and the bound no longer follows from the maps alone, so the other two
benefits remain but the free operator norm does not.

## Non-Cartesian Gradient Delay Correction

Eddy currents and gradient hardware timing delays displace non-Cartesian trajectory samples from their nominal positions, causing blurring and ring artifacts in radial and spiral acquisitions.

Both estimators are auto-calibrated from the radial data itself. `OpposingSpokes` reads the delay
from the shift between antiparallel spokes, which sample the same line in opposite directions
(Peters et al. 2003). `RING` (Rosenzweig et al. 2019) uses the fact that ideal spokes all cross at
$k = 0$: with anisotropic delays the pairwise intersection points move, and fitting them gives the
full 2D delay tensor (three parameters) from as few as three spokes.

```@docs
GradientDelay
OpposingSpokes
RING
estimate_gradient_delays
correct_gradient_delays
```

### When to use:
- Radial projection acquisitions (such as golden-angle or 3D stack-of-stars) suffering from trajectory delay artifacts.
- Opposing spoke pair cross-correlation (`OpposingSpokes`) or spoke intersection analysis (`RING`).

## References

- Kellman, P., & McVeigh, E. R. (2005). *Image reconstruction in SNR units: A general method for SNR measurement.* Magnetic Resonance in Medicine, 54(6), 1439-1447. <https://doi.org/10.1002/mrm.20713> — [`prewhiten`](@ref).
- Buehrer, M., Pruessmann, K. P., Boesiger, P., & Kozerke, S. (2007). *Array compression for MRI with large coil arrays.* Magnetic Resonance in Medicine, 57(6), 1131-1139. <https://doi.org/10.1002/mrm.21237> — [`SVDCompression`](@ref).
- Huang, F., Vijayakumar, S., Li, Y., Hertel, S., & Duensing, G. R. (2008). *A software channel compression technique for faster reconstruction with many channels.* Magnetic Resonance Imaging, 26(1), 133-141. <https://doi.org/10.1016/j.mri.2007.04.010>
- Zhang, T., Pauly, J. M., Vasanawala, S. S., & Lustig, M. (2013). *Coil compression for accelerated imaging with Cartesian and non-Cartesian sampling.* Magnetic Resonance in Medicine, 69(2), 571-582. <https://doi.org/10.1002/mrm.24267> — [`GeometricCompression`](@ref).
- Uecker, M., et al. (2014). *ESPIRiT — an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA.* Magnetic Resonance in Medicine, 71(3), 990-1001. <https://doi.org/10.1002/mrm.24751> — [`ESPIRiT`](@ref).
- Peters, D. C., Derbyshire, J. A., & McVeigh, E. R. (2003). *Centering the projection reconstruction trajectory: Reducing gradient delay errors.* Magnetic Resonance in Medicine, 50(1), 1-6. — [`OpposingSpokes`](@ref).
- Rosenzweig, S., Holme, H. C. M., & Uecker, M. (2019). *Simple auto-calibrated gradient delay estimation from few spokes using radial intersections (RING).* Magnetic Resonance in Medicine, 81(3), 1898-1906. <https://doi.org/10.1002/mrm.27506> — [`RING`](@ref).

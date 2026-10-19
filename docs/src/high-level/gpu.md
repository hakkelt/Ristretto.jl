# GPU Reconstruction

A reconstruction runs where its k-space lives. There is no `device` keyword: move the
acquisition to a GPU and `reconstruct` builds every operator for the GPU, runs the solver
there, and returns the image as a device array.

```julia
using Ristretto
using Adapt, CUDA

acq_gpu = adapt(CuArray, acq)           # k-space, sensitivity maps and dcf move to the GPU
img_gpu = reconstruct(acq_gpu, IterativeReconstruction(TotalVariation2D(1e-3)))
img = Array(img_gpu)                    # back to host memory
```

Any GPU array type that implements GPUArrays.jl works the same way (`CuArray`, `ROCArray`,
`MtlArray`, …); loading `GPUArrays` and `KernelAbstractions` — which every GPU backend does —
loads the package's GPU extension. `adapt(Array, acq_gpu)` moves an acquisition back.

`adapt` keeps `NamedDimsArray` names. It leaves the subsampling pattern and a non-Cartesian
trajectory on the host: the sampling operators take host indices and the NFFT plan takes a
host trajectory, and both are small.

## Precision

Use `ComplexF32` data. Consumer GPUs run double precision at a small fraction of their single
precision rate, and every result below was checked in single precision against the CPU (relative
differences of 1e-7 to 1e-4, depending on the number of iterations).

## What runs on the device

| Part | On a GPU |
|---|---|
| Cartesian encoding (FFT, sensitivity maps, subsampling) | native |
| Non-Cartesian encoding (NFFT, Toeplitz normal operator) | native |
| Scaling rules | native; only the sample a quantile reads is copied to the host |
| All proximal algorithms (FISTA, POGM, ADMM, CG, …) | native |
| L1/L2/L0 image and wavelet terms, TV, anisotropic TV, second-order TV, TGV, edge-preserving roughness, temporal TV, temporal Fourier, joint sparsity, constraints | native |
| `LowRank`, `RankLimit` | native (the device's SVD) |
| `LocallyLowRank`, `MultiScaleLowRank` | native block gathering; on CUDA all blocks are thresholded at once by CUSOLVER's batched Jacobi solvers whenever a block has at most 32 voxels or 32 frames (5-25x faster than the fallback on an A100); otherwise each block goes through its small Gram matrix, whose eigendecomposition runs on the host |
| `StructuredLowRank` (`:c`, `:s`, `:g`, ALOHA weights) | native lifts; the device's SVD |
| `L1Contourlet` | the transform runs on the host, the rest on the device |
| Direct, Homodyne, POCS, phase-constrained partial Fourier | native |
| GRAPPA, SPIRiT | on a host copy; the image is moved back |
| Sensitivity estimation, coil compression, density compensation, gradient delay correction | on a host copy; the result is moved back |
| Prewhitening, sensitivity normalization | native |
| Simulation | host only; `adapt` its result |

A `PlugAndPlay` denoiser receives device arrays and has to handle them.

## Settings that differ on a device

- `threaded` is ignored: the device kernels are the parallelism, and a host thread per slice
  would only queue work on the same device.
- `task_executor = MultiThreadingExecutor()` is rejected for the same reason.
- `disable_task_splitting` defaults to [`DEVICE_DISABLES_TASK_SPLITTING`](@ref Ristretto.DEVICE_DISABLES_TASK_SPLITTING)
  (see below). Pass `false` to split anyway.
- `fft_planning` has no effect: the device FFT is not FFTW.

### [Task splitting](@id gpu-task-splitting)

On the host, independent slices are solved in parallel. On a device they would be solved one
after another, each paying its own kernel launches, operator build and scaling pass, while one
solve over the whole stack runs the same arithmetic in larger kernels. Task splitting is
therefore off by default on a device.

Measured on an NVIDIA A100 (20 iterations, 8–16 coils, `benchmark/gpu_task_splitting.jl`),
the time of the split solve divided by the time of the unsplit one:

| case | image | batch | CG-SENSE | TV |
|---|---|---|---|---|
| multi-slice | 64² | 16 slices | 1.31 | 1.44 |
| multi-slice | 128² | 16 slices | 0.99 | 1.23 |
| multi-slice | 256² | 16 slices | 1.15 | 1.28 |
| multi-slice | 384² | 8 slices | 0.94 | 0.98 |
| cine | 128² | 30 frames | 4.23 | 1.42 |

Splitting only breaks even once a single slice fills the device. Pass
`disable_task_splitting = false` to split anyway.

## Mixing host and device data

The k-space, the sensitivity maps and the density compensation of one acquisition must all be in
host memory or all in device memory, and so must an initial guess `x₀`; a mismatch throws an
`ArgumentError` naming both array types. Move them together with `adapt`.

```@docs
Ristretto.DEVICE_DISABLES_TASK_SPLITTING
```

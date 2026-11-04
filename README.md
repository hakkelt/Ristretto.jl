# Ristretto.jl

[![CI](https://github.com/hakkelt/Ristretto.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/hakkelt/Ristretto.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://hakkelt.github.io/Ristretto.jl/dev/)
[![Codecov](https://codecov.io/gh/hakkelt/Ristretto.jl/graph/badge.svg)](https://codecov.io/gh/hakkelt/Ristretto.jl)

**Ristretto** (*Regularized Imaging Solvers Toolbox: Rapid, Efficient, Threaded, Tunable, Open*)
reconstructs MRI images from k-space data: Cartesian and non-Cartesian, single- and multi-coil,
2D, 3D and dynamic, from a direct adjoint through GRAPPA, SPIRiT and partial Fourier to
compressed sensing, low-rank, low-rank + sparse and calibrationless structured low-rank models,
on the CPU or a GPU. One call, `reconstruct(acq, method)`, covers the common cases; the operators
and optimization problems underneath stay open for your own.

## Installation

Install Julia with [juliaup](https://github.com/JuliaLang/juliaup)
(`curl -fsSL https://install.julialang.org | sh` on Linux and macOS,
`winget install --name Julia --id 9NJNWW8PVKMN -e` on Windows), then:

```julia
using Pkg
Pkg.add("Ristretto")
```

## Examples

`kspace` is `(kx, ky, coil)` k-space, `smaps` the matching sensitivity maps, `mask` the acquired
phase encodes; an MRD file read with MRIFiles goes in as `AcquisitionInfo(raw)`.

```julia
using Ristretto

# Direct: zero-filled k-space through the adjoint, coils combined with the maps.
acq = AcquisitionInfo(kspace; sensitivity_maps = smaps, subsampling = (:, mask))
img = reconstruct(acq)

# SENSE: the unregularized least-squares problem, solved by conjugate gradients.
img = reconstruct(acq, IterativeReconstruction(; maxit = 30))

# Compressed sensing: ℓ₁-wavelet and total-variation penalties.
img = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(2.0f-3), TotalVariation2D(1.0f-3); maxit = 50))
```

The result is a `ReconImage` carrying the acquisition's geometry, which FileIO's `save` writes to
NIfTI, DICOM or MRD (`save("recon.nii", img)`). Underneath, the encoding operator and a
[StructuredOptimization](https://github.com/JuliaFirstOrder/StructuredOptimization.jl) problem
are a few lines away:

```julia
using Ristretto: get_encoding_operator
using Ristretto.StructuredOptimization
using Ristretto.WaveletOperators: WaveletOp

A = get_encoding_operator(acq)                            # mask ∘ FFT ∘ coil maps
W = WaveletOp(ComplexF32, wavelet(WT.db4), size(smaps)[1:2])
# The variable being optimized, initialized with the adjoint (zero-filled) image A'y.
x = Variable(A' * acq.kspace_data)
@minimize ls(A * x - acq.kspace_data) + 2.0f-3 * norm(W * x, 1) with FISTA(maxit = 50)
img = ~x
```

## Performance

Benchmarks against BART, SigPy, MRIReco.jl, MIRT.jl and MRpro (time to a common accuracy, matched
effort, thread scaling and GPU) are published in
[RistrettoBenchmarking](https://hakkelt.github.io/RistrettoBenchmarking/), dated and with the
version of every toolbox measured.

## Citation

```bibtex
@software{Ristretto,
  author = {Hakkel, Tamás},
  title = {Ristretto.jl: Regularized Imaging Solvers Toolbox for MRI reconstruction in Julia},
  year = {2026},
  url = {https://github.com/hakkelt/Ristretto.jl}
}
```

## License

MIT, see [LICENSE.txt](LICENSE.txt).

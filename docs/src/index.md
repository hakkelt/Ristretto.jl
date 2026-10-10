# Ristretto.jl

*Regularized Imaging Solvers Toolbox: Rapid, Efficient, Threaded, Tunable, Open*

Ristretto reconstructs MRI images from k-space data: Cartesian and non-Cartesian, single- and
multi-coil, 2D, 3D and dynamic, from a direct adjoint to compressed sensing, low-rank and
structured low-rank models. One call, `reconstruct(acq, method)`, covers the common cases, and
the operators and optimization problems underneath are open for custom reconstructions.

## Installation

Install Julia with [juliaup](https://github.com/JuliaLang/juliaup):

```sh
curl -fsSL https://install.julialang.org | sh    # Linux and macOS
winget install --name Julia --id 9NJNWW8PVKMN -e # Windows
```

The package is not registered yet. The versions of AbstractOperators, OperatorCore,
ProximalOperators, ProximalAlgorithms and StructuredOptimization it needs are still under review
upstream, so they ship inside the package. Install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/hakkelt/Ristretto.jl")
```

## A first reconstruction

```julia
using Ristretto

acq = AcquisitionInfo(kspace; sensitivity_maps)            # k-space: (kx, ky, coil)
x_direct = reconstruct(acq)                                # adjoint (zero-filled SENSE)
x_cs = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(2.0f-3); maxit = 60))
```

[Getting started](tutorials/01_getting_started.md) runs this on a simulated acquisition, with
pictures.

## Where to read on

- **Tutorials** walk through the package topic by topic, from a first reconstruction to real
  scanner data. Each one runs as a page here and downloads as a Jupyter notebook.
- **High-level interface** is the reference for each part of `reconstruct`: acquisition data,
  preprocessing, simulation, methods, regularizers, solvers, export.
- **Low-level interface** covers the operators and the optimization problems directly, for
  reconstructions the high-level interface does not build.
- **Theoretical background** states the models and the conventions.
- **Related packages** compares Ristretto with other reconstruction toolboxes and places it in
  the Julia MRI ecosystem.

## The API surface

`using Ristretto` brings in the names a user needs to assemble a reconstruction from the built-in
pieces: the regularization terms, the reconstruction methods, the configuration types and the
top-level verbs `reconstruct`, `build_model`, `simulate_acquisition` and friends.

Everything needed to *extend* the package — the abstract supertypes you subtype, and the interface
functions you add methods to (`get_operator`, `materialize`, `get_encoding_operator`, …) — is public
and documented, but deliberately not exported. Import those explicitly:

```julia
using Ristretto: Regularization, get_operator, materialize
```

The package also does not reexport its dependencies. Code that builds operators or optimization
problems by hand needs its own `using Ristretto.AbstractOperators` /
`using Ristretto.StructuredOptimization`.

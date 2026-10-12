# Image Decomposition

*Tutorial: [Dynamic imaging and decomposition](../tutorials/07_dynamic_and_decomposition.md).*

Image decomposition models the reconstructed image as a sum of additive
components, each with its own regularizer — the canonical example being
low-rank + sparse (L+S) decomposition of dynamic MRI. This is a different
concept from [Task Splitting](task_splitting.md), which splits a
*single-image* problem over independent batch dimensions (e.g. slices); the
two can be combined (see [Interaction with Task Splitting](@ref
image-decomposition-task-splitting) below).

See [Theoretical Background](../theory.md#Additive-Image-Decomposition) for the
underlying optimization model.

## API Reference

```@docs
Component
components
total_image
drop_components
```

## Basic Usage

```@setup imgdecomp
using Ristretto
using Random
Random.seed!(123)
```

Declare each component with a name and one or more regularizations, then pass
the components to `IterativeReconstruction`:

```@example imgdecomp
using Ristretto

ksp = rand(ComplexF32, 64, 64, 4)
acq = AcquisitionInfo(ksp; is3D = false)

img = reconstruct(
    acq,
    IterativeReconstruction(
        Component(:smooth, L2Image(0.01)),
        Component(:sparse, L1Image(0.05)); maxit = 30); verbosity = Silent())

println(typeof(img))
println("Components: ", keys(components(img)))
```

The result is a [`ReconImage`](@ref) whose array is the sum of the components,
which it also holds:

```@example imgdecomp
using LinearAlgebra

sum(values(components(img))) ≈ total_image(img)
```

Individual components stay accessible via `.components`, or, more concisely,
directly as a property — `img.smooth` is shorthand for `img.components.smooth`:

```@example imgdecomp
img.components.smooth isa AbstractArray
img.smooth isa AbstractArray
```

The components are plain arrays shaped like the image, which the image's
[`header`](@ref Ristretto.header) describes. Keyword indexing selects the same part of
every component, so `img[time = 3].smooth` is frame 3 of the smooth component. Once the
components are no longer needed, `drop_components(img)` returns the image without them, so
their memory can be released. The image's own properties (`data`, `header`,
`components`) always resolve first, so a component cannot be named after one of
them — `reconstruct` (via `Component`/`check_components`) and the `ReconImage`
constructor both reject that collision, since such a component would otherwise
be unreachable through dot access:

```@example imgdecomp
try
    ReconImage(zeros(2, 2); components = (header = zeros(2, 2),))
catch e
    println(e)
end
```

`propertynames(img)` lists both the image's own properties and every component name, and
accessing an unknown property raises an `ArgumentError` naming the available
ones:

```@example imgdecomp
println(propertynames(img))
try
    img.nonexistent
catch e
    println(e)
end
```

To get a plain array of the sum, without the header and the components, use
`Array`:

```@example imgdecomp
x = Array(img)
println(typeof(x))
```

## Low-Rank + Sparse (L+S)

The model image decomposition was built for is the L+S decomposition of dynamic
MRI (Otazo, Candès & Sodickson, *Magn Reson Med* 2015): a low-rank component
`L` carrying the temporally correlated background, plus a sparse component `S`
carrying the dynamic foreground.

```julia
img = reconstruct(
    acq_dynamic,
    IterativeReconstruction(
        Component(:lowrank, LowRank(5e-2; time_dim = 3)),
        Component(:sparse, TemporalTotalVariation(2e-2; time_dim = 3)); maxit = 100))

background = img.lowrank   # e.g. static anatomy
dynamics   = img.sparse    # e.g. contrast uptake, motion
```

Common choices for the sparse component are [`TemporalTotalVariation`](@ref)
(irregular dynamics), [`L1TemporalFourier`](@ref) (periodic dynamics, the
original k-t SPARSE transform) or [`L1Image`](@ref); the low-rank component is
[`LowRank`](@ref), or [`LocallyLowRank`](@ref) when the dynamics vary across
the field of view.

!!! note "Solvable combinations"
    - **L+S dynamic MRI**: `LowRank` (or `LocallyLowRank`) + `TemporalTotalVariation`
      (or `L1TemporalFourier` or `L1Image`) cleanly separates background from motion/contrast.
    - **Infimal convolution TV**: `Component(:cartoon, TotalVariation2D(λ))` +
      `Component(:ramp, SecondOrderTotalVariation2D(λ))` splits the image into a
      piecewise-constant and a piecewise-linear part.
    - **Multi-scale low rank**: one [`LocallyLowRank`](@ref) component per block
      size gives the exact model of Ong & Lustig, with the scales separated;
      [`MultiScaleLowRank`](@ref) is the single-image approximation of the same idea.

## What Additive Components Are Not

Every component is an image that is *summed into the data term*: the model is
`‖E(x₁ + x₂ + …) - y‖²`. This is the right structure for L+S, for
infimal-convolution-style splittings of an image into parts with different
regularity, and for background/foreground separation.

It is *not* a general auxiliary-variable mechanism. Regularizers such as total
generalized variation introduce an auxiliary variable that is coupled to the
image through a term like `‖∇x - w‖`, while being absent from the data term
entirely. That variable is not an additive image component, so it does not
follow from the `Component` API. [`TotalGeneralizedVariation2D`](@ref) therefore
uses a separate mechanism — a regularization may declare auxiliary variables of
its own, which are solved for alongside the image and discarded afterwards — and
is used like any other regularizer, not as a component.

Two related models *are* expressible as components, because they really are
additive splittings:

- **Infimal-convolution TV**: `Component(:cartoon, TotalVariation2D(λ))` plus
  `Component(:ramp, SecondOrderTotalVariation2D(λ))` splits the image into a
  piecewise-constant and a piecewise-linear part.
- **Multi-scale low rank**: one [`LocallyLowRank`](@ref) component per block
  size gives the exact model of Ong & Lustig, with the scales separated;
  [`MultiScaleLowRank`](@ref) is the single-image approximation of the same idea.

## At Least Two Components

Image decomposition requires **at least two** components — a single component
is rejected, since a one-component reconstruction is just the plain
regularization API:

```@example imgdecomp
try
    reconstruct(acq, IterativeReconstruction(Component(:only, L1Image(0.05))); verbosity = Silent())
catch e
    println(e)
end
```

Component names must also be unique.

## Multiple Regularizations per Component

A `Component` can combine several regularizations, exactly like the plain
regularization API:

```@example imgdecomp
img_multi = reconstruct(
    acq,
    IterativeReconstruction(
        Component(:structured, L1Wavelet2D(0.01), TotalVariation2D(0.005)),
        Component(:sparse, L1Image(0.05)); maxit = 20); verbosity = Silent())
nothing # hide
```

## Choosing λ per Component

Each component's regularization strength is set independently, exactly as for
the plain regularization API — there is no automatic balancing between
components. As a starting point, scale `λ` for each regularizer the same way
you would if that component were reconstructed on its own (see
[Regularization](regularization.md)), then adjust based on how much of the
signal each component should absorb.

## Initial Guess

By default, the first component is initialized with the direct (adjoint)
reconstruction and the remaining components start at zero — the standard
L+S/RPCA warm start. Override this with `x₀` as a `Tuple` (component order) or
`NamedTuple` (by component name):

```@example imgdecomp
x̂ = reconstruct(acq; verbosity = Silent())
img_warm = reconstruct(
    acq,
    IterativeReconstruction(
        Component(:smooth, L2Image(0.01)),
        Component(:sparse, L1Image(0.05)); maxit = 30); x₀ = (smooth = x̂, sparse = zero(x̂)), verbosity = Silent())
nothing # hide
```

## Solver Applicability

The same rules that govern algorithm selection for a single image apply here,
term by term: a component with one regularization whose operator is
`is_AAc_diagonal` (e.g. `L1Image`, `L1Wavelet2D/3D`) can be solved with
FISTA/PANOC-family algorithms; a component with multiple regularizations, or a
non-tight operator, falls back to ADMM — exactly as for multiple
regularizations on a single image. Use `StructuredOptimization.print_diagnostics`
or `suggest_algorithm` to see which condition failed if a forced algorithm
errors.

## Performance Notes

- The data term applies the encoding operator `𝒜` to the *sum* of the
  component variables (`𝒜*(x₁ + x₂ + …)`), not once per component, so its
  cost matches a single-image reconstruction with the same `𝒜`.
- The normal-operator substitution ($\mathcal{A}^*\mathcal{A}$) applies here as it does to a
  single-image reconstruction: the data term is the same plain least-squares term, and whether
  its gradient goes through the normal operator is decided when the problem is parsed, against
  the joint domain of every component variable.
- The Lipschitz constant of the data term scales with the number of
  components (for `n` components sharing a unit-norm operator `𝒜`,
  `‖[𝒜 … 𝒜]‖ = √n‖𝒜‖`), so `reconstruct` defaults `Lf = n_components` for
  FISTA/PANOC-family algorithms when reconstructing with components (instead
  of `Lf = 1` for a single image). Overriding `Lf` explicitly on the algorithm
  bypasses this default.

## [Interaction with Task Splitting](@id image-decomposition-task-splitting)

Image decomposition composes with [Task Splitting](task_splitting.md):
if the data has batch dimensions (e.g. slices) that none of the components'
regularizations couple, `reconstruct` still splits the task over those
dimensions automatically, solving each slice's image-decomposition problem
independently and stacking both the total image and each component:

```@example imgdecomp
nx, ny, nslices, nc = 32, 32, 3, 2
ksp_ms = rand(ComplexF32, nx, ny, nc, nslices)
acq_ms = AcquisitionInfo(ksp_ms; is3D = false)

img_ms = reconstruct(
    acq_ms,
    IterativeReconstruction(
        Component(:smooth, L2Image(0.01)),
        Component(:sparse, L1Image(0.05)); maxit = 10); verbosity = Silent())
println(size(img_ms))
println(size(img_ms.smooth))
```

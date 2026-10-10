# ProximalOperators.jl Summary

!!! note "Bundled package"
    Ristretto ships its own copy of ProximalOperators.jl and ProximalAlgorithms.jl. Import them
    through Ristretto, as `using Ristretto.ProximalOperators` and `using
    Ristretto.ProximalAlgorithms`, and do not `Pkg.add` the registered packages: the bundled
    version carries work that is not registered yet (multithreading, GPU support, new operators,
    functions and algorithms). Upstreaming it is under way, and the bundled copy goes away once
    the registered releases have it. Documentation of the bundled version:
    [ProximalOperators.jl (fork)](https://hakkelt.github.io/ProximalOperators.jl/dev/),
    [ProximalAlgorithms.jl (fork)](https://hakkelt.github.io/ProximalAlgorithms.jl/dev/).

This page gives a quick overview of the `ProximalOperators.jl` package, which supplies a large catalog of convex (and some nonconvex) function objects together with their proximal operators (and gradients when available). These building blocks can be used by (`ProximalAlgorithms.jl`) and by the reconstruction routines in this toolbox to express regularization terms and constraints.

## Goal

Provide efficient, composable implementations of
\( f: \mathbb{C}^n \to \mathbb{R}\cup\{+\infty\} \)
with:
- Fast evaluation of value `f(x)`
- (Optionally) gradient evaluation `gradient(f, x)` / `gradient!(y, f, x)`
- Proximal mapping `prox(f, x, γ)` / in-place `prox!(y, f, x, γ)`
- Traits advertising structural properties (convexity, separability, smoothness) exploited by algorithms.

## Most Common Norm / Regularization Functions

The following constructors (all exported) cover the majority of regularization needs in MRI reconstruction and sparse inverse problems:

| Function | Description | Typical Use |
|----------|-------------|-------------|
| `NormL0(λ)` | Counts nonzeros (scaled) | Hard sparsity (often surrogate) |
| `NormL1(λ)` | \( λ\|x\|_1 \) (weighted variant) | Soft sparsity / wavelets |
| `NormL2(λ)` | \( λ\|x\|_2 \) | Magnitude penalization / normalization |
| `SqrNormL2(λ)` | \( \tfrac{λ}{2}\|x\|_2^2 \) (ridge) | L2Image / quadratic penalization |
| `NormL21(λ)` | Group sparsity (sum of L2 norms over groups) | Multidimensional total variation |
| `NormLinf(λ)` | \( λ\|x\|_\infty \) | Robust range control |
| `ElasticNet(λ1, λ2)` | \( λ_1\|x\|_1 + \tfrac{λ_2}{2}\|x\|_2^2 \) | Combined sparsity + shrinkage |
| `TotalVariation1D(λ)` | Discrete TV along 1D axis | Temporal / 1D smoothing |
| `NuclearNorm(λ)` | \( λ\|X\|_* \) (sum singular values) | Low-rank models (dynamic MRI) |
| `CubeNormL2(λ)` | \( λ\|x\|_2^3 \) | Specialized smoothing |
| `IndBox(lower, upper)` | Indicator of box constraints | Hard constraints on values |
| `IndNonnegative()` | Indicator of non-negativity | Enforce non-negative solutions |
| `IndBallRank(r)` | Indicator of rank ≤ r | Hard low-rank constraints |

(See the full list in the upstream documentation for additional indicators, losses, and composite penalties.)

## Quick Usage Examples

```@example
using Ristretto.ProximalOperators
x = randn(10)

# L1 norm
f1 = NormL1(0.5)
val1 = f1(x)
# Prox (soft-threshold)
y1, fy1 = prox(f1, x, 0.2)

# Elastic net
fen = ElasticNet(0.3, 0.8)
val_en = fen(x)
y_en, fy_en = prox(fen, x, 0.5)

# Group sparsity (split into groups of size 2)
# NormL21 expects a partition; simplest is reshape
xg = reshape(randn(12), 2, 6)  # 6 groups of length 2
fg = NormL21(0.4)
val_g = fg(xg)
yg, fyg = prox(fg, xg, 0.3)

println("L1 value: ", val1)
println("Elastic net prox value: ", fy_en)
println("Group sparsity prox value: ", fyg)
```

## Implementing a Custom Norm (Minimal Example)

Below is a minimal implementation of L2-norm:
```math
f(x) = \|x\|_2 = \sqrt{\sum_{i=1}^n |x_i|^2}
```

Key steps:
1. Define a `struct` holding parameters.
2. Add trait methods (used by StructuredOptimization.jl to parse terms for algorithms).
3. Define call `(f)(x)` for value.
4. Implement `gradient!(y, f, x)` if differentiable (not here).
4. Implement `prox!(y, f, x, γ)` (and optionally for complex or array γ).

```@example proxops
import ProximalCore: is_proximable, is_separable, is_convex, is_locally_smooth, prox!, gradient!
using Ristretto.ProximalOperators
using LinearAlgebra: norm

struct MyNormL2{R}
    lambda::R
    function MyNormL2{R}(lambda::R) where R
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_convex(f::Type{<:MyNormL2}) = true
is_positively_homogeneous(f::Type{<:MyNormL2}) = true
is_locally_smooth(f::Type{<:MyNormL2}) = true

MyNormL2(lambda::R=1) where R = MyNormL2{R}(lambda)

(f::MyNormL2)(x) = f.lambda * norm(x)

function prox!(y, f::MyNormL2, x, gamma)
    normx = norm(x)
    scale = max(0, 1 - f.lambda * gamma / normx)
    for i in eachindex(x)
        y[i] = scale*x[i]
    end
    return f.lambda * scale * normx
end

function gradient!(y, f::MyNormL2, x)
    fx = norm(x) # Value of f, without lambda
    if fx == 0
        y .= 0
    else
        y .= (f.lambda / fx) .* x
    end
    return f.lambda * fx
end

# Test the custom norm
x = randn(8)
f_custom = MyNormL2(0.7)
val_before = f_custom(x)
yc, val_after = prox(f_custom, x, 0.4)
println("Custom norm value (before): ", val_before)
println("Custom norm value (after prox): ", val_after)
```

## Proximable Functions Defined by This Package

Some regularization terms need a proximal operator that `ProximalOperators.jl` does not
provide. These are defined here and used through the corresponding regularization types
rather than directly.

```@docs
Ristretto.BlockNuclearNorm
Ristretto.DenoiserProx
```

Two of them started here and were moved upstream, because nothing about either is MRI-specific:
`MultiScaleLowRank` combines its per-scale block nuclear norms with the proximal average, and
`HardConsistency` projects onto ``\{x : \mathcal{A}x = y\}`` with a matrix-free affine indicator.

```@docs
ProximalOperators.ProximalAverage
ProximalOperators.IndAffine
ProximalOperators.IndAffineCG
```

What stayed in this package for the hard-consistency constraint is the one MRI-specific part: knowing
that `is_AAc_diagonal(𝒜)` makes the projection a division, which `hard_consistency_prox` passes to
`IndAffineCG` as its `AAc_diag`.

## See Also

- Upstream docs: [https://juliafirstorder.github.io/ProximalOperators.jl/stable/](https://juliafirstorder.github.io/ProximalOperators.jl/stable/)
- Algorithm layer: [`ProximalAlgorithms.jl`](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
- This toolbox high-level regularization page: [Regularization](@ref)

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

`ProximalOperators.jl` supplies function objects with their value `f(x)`, proximal map
`prox(f, x, γ)` / `prox!(y, f, x, γ)`, gradient where it exists, and traits (convexity,
separability, smoothness) that the solvers read. Every regularizer in Ristretto materializes to
one of them. [The low-level interface](../tutorials/12_low_level_interface.md) calls them
directly (§5) and implements a new one (§8).

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

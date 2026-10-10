# Optimization Algorithms

*Tutorial: [Algorithms and configuration](../tutorials/06_algorithms_and_configuration.md).*

Ristretto supports multiple iterative optimization algorithms for solving MRI reconstruction problems. This guide helps you choose and configure the right algorithm for your needs.

## Quick Algorithm Selection

**Not sure which to use?** Let `reconstruct()` choose automatically:

```julia
img = reconstruct(acq, IterativeReconstruction(regularization))
# Automatically selects appropriate algorithm from DEFAULT_ALGORITHMS
```

**How does it decide?** Here's a decision tree:

```
Is your problem smooth (no L1, TV, etc.)?
├─ Yes → Is every term quadratic (data term, Tikhonov)?
│   ├─ Yes → Use CGNR (Conjugate Gradient Normal Residual)
│   └─ No → Use LBFGS (an edge-preserving roughness penalty, say)
└─ No → Does it have a single non-smooth regularizer where the wrapped operator is symmetric* (e.g. wavelets, temporal Fourier)?
    ├─ Yes → Use POGM (Proximal Optimized Gradient Method)
    └─ No → Use ADMM (Alternating Direction Method of Multipliers)
```

*Symmetric means the operator satisfies `E' * E = E * E'`, e.g. Fourier-based operators. But actually a more loose condition (`is_AAc_diagonal` from `OperatorCore.jl`) is used: `E' * E = diag(d)` for some `d`, i.e. the normal operator is equal to element-wise scaling.

## ProximalAlgorithms.jl Interface

Ristretto builds on [ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl). All algorithms from that package can be used directly. There are four recommended algorithms for MRI reconstruction used by default in `reconstruct()`:
- `CGNR`: Conjugate Gradient Normal Residual for quadratic problems
- `LBFGS`: limited-memory BFGS for smooth, non-quadratic problems (`NCG` solves the same problems)
- `POGM`: Proximal Optimized Gradient Method for a single non-smooth regularizer (`FISTA` solves the
  same problems and is one `algorithm = FISTA()` away)
- `ADMM`: Alternating Direction Method of Multipliers for multiple regularizers

All algorithms from this library share a common interface with the following parameters:
* `maxit::Int`: maximum number of iteration
* `stop::Function`: termination condition, `stop(::T, state)` should return `true` when to stop the iteration
* `solution::Function`: solution mapping, `solution(::T, state)` should return the identified solution
* `verbose::Bool`: whether the algorithm state should be displayed
* `freq::Int`: every how many iterations to display the algorithm state
* `summary::Function`: function returning a summary of the iteration state, `summary(k::Int, iter::T, state)` should return a vector of pairs `(name, value)`
* `display::Function`: display function, `display(k::Int, alg, iter::T, state)` should display a summary of the iteration state

## Default Algorithms

```@setup imports
using Ristretto
using GeometricMedicalPhantoms
using MIRTjim: jim
using Plots
using Random

Random.seed!(0)

x = create_shepp_logan_phantom(128, 128, :axial; ti = MRISheppLoganIntensities(), eltype = ComplexF32);
x_noisy = x + 0.02f0 * randn(ComplexF32, 128, 128);
smaps = coil_sensitivities(128, 128, 8);
acq_full = AcquisitionInfo(
    image_size=(128, 128)
)
data_full = simulate_acquisition(x_noisy, acq_full; inverse_crime_check = false, keep_sensitivity_maps = true)

pdf = VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.05)
pattern = create_sampling_pattern(pdf, (128, 128))
acq = AcquisitionInfo(
    image_size=(128, 128),
    subsampling=pattern,
    sensitivity_maps=smaps
)
data = simulate_acquisition(x_noisy, acq; inverse_crime_check = false, keep_sensitivity_maps = true)
```

### Conjugate Gradient Normal Residual (CGNR)

This algorithm solves linear systems of the form

	argminₓ ‖Ax - b‖₂² + ‖λx‖₂² 

where `A` is a symmetric positive definite linear operator, and `b` is the measurement vector,
and `λ` is the L2 regularization parameter. `λ` might be scalar or an array of the same size
as `x`. If `λ` is zero, the problem reduces to a least-squares problem:

	argminₓ ‖Ax - b‖₂²

**Best for:** Least-squares problems with optional L2Image regularization

**Properties:**
- Solves normal equations: A'A·x = A'·b
- Good for ill-conditioned problems
- Variant of CG with different mathematical properties

**Parameters:**
- `λ=0`: L2 regularization parameter (default: 0)
- `P`: preconditioner (optional)
- `P_is_inverse`: whether `P` is the inverse of the preconditioner (default: `false`)

**References:**
1. Hestenes, M.R. and Stiefel, E., "Methods of conjugate gradients for solving linear systems."
   Journal of Research of the National Bureau of Standards 49.6 (1952): 409-436.

**Pros:**
- ✅ Very fast convergence for appropriate problems
- ✅ No hyperparameter tuning
- ✅ Memory efficient

**Cons:**
- ❌ Only for a small class of problems
- ❌ Can't handle L1, TV, or other non-smooth terms

**Example:**
```@example imports
reconstruct(data, IterativeReconstruction(L2Image(1e-4); algorithm = CGNR(), maxit = 2); verbosity = Silent()) # hide
GC.gc() # hide
img = reconstruct(data, IterativeReconstruction(L2Image(1e-4); algorithm = CGNR(), maxit = 20));
nothing # hide
```

#### Preconditioning

Passing `P` to `CG` or `CGNR` switches the solve to the preconditioned variant, which converges in
fewer iterations whenever `𝒜ᴴ𝒜` is badly conditioned. It changes the *path*, not the solution: both
variants minimize the same objective, so the thing to measure is the iteration count at a given
error, not the error at convergence.

`P` is applied as `z = P \ r` by default. An `AbstractOperator` supports `mul!` but not `ldiv!`, so
pass the **inverse** preconditioner and set `P_is_inverse = true`; a `Diagonal`, a factorization, or
anything else that implements `ldiv!` can be passed directly with `P_is_inverse = false`.

The natural choice for SENSE is the diagonal image-domain approximation of `𝒜ᴴ𝒜`, the coil coverage
`Σ_c |S_c|² + λ`:

```@example imports
using Ristretto.AbstractOperators: DiagOp
coverage = real(sum(abs2, unname(smaps); dims = 3)[:, :, 1])
P⁻¹ = DiagOp(ComplexF32.(1 ./ (coverage .+ 1.0f-4)))
img_pc = reconstruct(
    data,
    IterativeReconstruction(
        L2Image(1e-4); algorithm = CGNR(; P = P⁻¹, P_is_inverse = true), maxit = 20
    );
    verbosity = Silent(),
);
nothing # hide
```

### Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

This algorithm solves convex optimization problems of the form

    minimize f(x) + g(x),

where `f` is smooth.

**Best for:** Single non-smooth regularizer (e.g. L1) with symmetric operator (e.g. wavelets)

!!! note
    FISTA is only an alias for FastForwardBackward from ProximalAlgorithms.jl.

**Properties:**
- Accelerated gradient method
- Handles non-smooth regularization
- Faster convergence than basic ISTA

**Parameters:**
- `mf=0`: convexity modulus `f` (the smooth part of the objective, usually data fidelity term)
- `Lf=nothing`: Lipschitz constant of the gradient of `f` (usually equals to 1 because the Lipschitz contant of squared L2 norm is 1 and `get_encoding_operator` returns a normalized operator)
- `gamma=nothing`: stepsize, defaults to `1/Lf` if `Lf` is set, and `nothing` otherwise.
- `adaptive=true`: makes `gamma` adaptively adjust during the iterations; this is by default `gamma === nothing`.
- `minimum_gamma=1e-7`: lower bound to `gamma` in case `adaptive == true`.
- `reduce_gamma=0.5`: factor by which to reduce `gamma` in case `adaptive == true`, during backtracking.
- `increase_gamma=1.0`: factor by which to increase `gamma` in case `adaptive == true`, before backtracking.
- `extrapolation_sequence=nothing`: sequence (iterator) of extrapolation coefficients to use for acceleration.

**References:**
1. Tseng, "On Accelerated Proximal Gradient Methods for Convex-Concave Optimization" (2008).
2. Beck, Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202 (2009).

**Pros:**
- ✅ Fast for single regularizer
- ✅ Proven convergence guarantees
- ✅ Accelerated compared to basic gradient descent

**Cons:**
- ❌ Only one regularizer
- ❌ Requires Lipschitz constant (usually auto-estimated)
- ❌ Can be sensitive to step size

**Example:**
```@example imports
reconstruct(data, IterativeReconstruction(L1Wavelet2D(5e-3); algorithm = FISTA(), maxit = 2); verbosity = Silent()) # hide
GC.gc() # hide
img = reconstruct(data, IterativeReconstruction(L1Wavelet2D(5e-3); algorithm = FISTA(), maxit = 100));
nothing # hide
```

### Optimized Gradient Method (POGM)

This algorithm solves the same problem class as FISTA,

    minimize f(x) + g(x),

where `f` is smooth, using Kim & Fessler's optimized gradient method with Gu et al.'s adaptive
restart in place of FISTA's fixed momentum sequence.

**Best for:** Single non-smooth regularizer, same problems FISTA targets, when a worst-case
convergence rate twice as tight as FISTA's is wanted.

**Properties:**
- Accelerated gradient method with a provably optimal worst-case rate among first-order methods
- Adaptive restart avoids the oscillation fixed-momentum methods show near convergence
- Not in `DEFAULT_ALGORITHMS`; select it explicitly with `algorithm = POGM()`

**Parameters:** the same `mf`, `Lf`, `gamma`, `adaptive`, `minimum_gamma`, `reduce_gamma`,
`increase_gamma` as FISTA.

**References:**
1. Kim, Fessler, "Optimized First-order Methods for Smooth Convex Minimization", Mathematical
   Programming, vol. 159, pp. 81-107 (2016).
2. Gu, Bo, Kim, Yin, Fessler, "Optimized Gradient Method with Adaptive Restart for Faster
   Smooth Convex Minimization" (2018).

**Pros:**
- ✅ Tighter worst-case convergence rate than FISTA
- ✅ Adaptive restart improves practical convergence near the optimum

**Cons:**
- ❌ Only one regularizer, same as FISTA
- ❌ Requires Lipschitz constant (usually auto-estimated)

**Example:**
```@example imports
reconstruct(data, IterativeReconstruction(L1Wavelet2D(5e-3); algorithm = POGM(), maxit = 2); verbosity = Silent()) # hide
GC.gc() # hide
img = reconstruct(data, IterativeReconstruction(L1Wavelet2D(5e-3); algorithm = POGM(), maxit = 100));
nothing # hide
```

### Alternating Direction Method of Multipliers (ADMM)

This algorithm solves optimization problems of the form

	minimize ½‖Ax - b‖²₂ + ∑ᵢ gᵢ(Bᵢx)

where:
- `A` is a linear operator
- `b` is the measurement vector
- `gᵢ` are proximable functions with associated linear operators `Bᵢ`

**Best for:** Multiple regularizers or complex constraints

**Properties:**
- Splits problem into simpler subproblems
- Handles multiple regularizers naturally
- Handles contraints with non-symmetric operators (e.g. total variation)

**Parameters:**
- `P=nothing`: preconditioner for CG (optional)
- `P_is_inverse=false`: whether `P` is the inverse of the preconditioner
- `eps_abs=0`: absolute tolerance for convergence
- `eps_rel=1`: relative tolerance for convergence
- `cg_tol=1e-6`: CG tolerance
- `cg_maxit=100`: maximum CG iterations
- `y0=nothing`: initial dual variables
- `z0=nothing`: initial auxiliary variables
- `penalty_sequence=nothing`: penalty sequence for adaptive rho updating. The following options are available:
  - `FixedPenalty(rho)`: fixed penalty sequence with specified rho values
  - `ResidualBalancingPenalty(rho; mu=10.0, tau=2.0)`: adaptive penalty sequence based on residual balancing [2]
  - `SpectralRadiusBoundPenalty(rho; tau=10.0, eta=100.0)`: adaptive penalty sequence based on spectral radius bounds [3]
  - `SpectralRadiusApproximationPenalty(rho; tau=10.0)`: adaptive penalty sequence based on spectral radius approximation [4]
  Note: rho can be specified either as the `rho` parameter or within the penalty sequence constructor, but not both.

In `reconstruct`, a `rho` given either way is relative to the curvature $\|\mathcal{A}\|^2$ of
the data term and is multiplied by it before the solve, so the same value means the same thing for
a Cartesian and a radial encoding; see [Operator norm, step size and λ](@ref).

The adaptive penalty parameter schemes are implemented through the penalty sequence types, 
following various strategies from the literature. See the individual penalty sequence types 
for their specific update rules and references.

**References:**
1. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1), 1-122.
2. He, B. S., Yang, H., & Wang, S. L. (2000). Alternating direction method with self-adaptive penalty parameters for monotone variational inequalities. Journal of Optimization Theory and applications, 106(2), 337-356.
3. Lorenz, D. A., & Tran-Dinh, Q. (2019). Non-stationary Douglas–Rachford and alternating direction method of multipliers: Adaptive step-sizes and convergence. Computational Optimization and Applications, 74(1), 67–92. https://doi.org/10.1007/s10589-019-00106-9
4. Mccann, M. T., & Wohlberg, B. (2024). Robust and Simple ADMM Penalty Parameter Selection. IEEE Open Journal of Signal Processing, 5, 402–420. https://doi.org/10.1109/OJSP.2023.3349115

**Pros:**
- ✅ Handles multiple regularizers
- ✅ Very robust and stable
- ✅ Good for complex problems

**Cons:**
- ❌ Slower than FISTA for single regularizer
- ❌ More parameters to tune
- ❌ Each iteration more expensive
- ❌ No convergence guarantees in general

**Example:**
```@example imports
# Multiple regularizers
reg = (L1Wavelet2D(5e-3), TotalVariation2D(1e-3))
reconstruct(data, IterativeReconstruction(reg...; algorithm = ADMM(), maxit = 2); verbosity = Silent()) # hide
GC.gc() # hide
img = reconstruct(data, IterativeReconstruction(reg...; algorithm = ADMM(), maxit = 50));
nothing # hide
```

### Douglas-Rachford Splitting (`DouglasRachford`)

**When to use:**
- Problems with two proximable terms, such as hard data consistency (`HardConsistency`) with an indicator or proximable regularizer (e.g. `NonNegative`, `BoxConstraint`, `L1Image`).
- Alternating projections between two convex sets / constraints.

**How it works:**
Douglas-Rachford splitting solves problems of the form ``\min f(x) + g(x)`` where both ``f`` and ``g`` have efficient proximal operators. It updates iterates via reflected proximal evaluations:
```math
y_{k+1} = \operatorname{prox}_{\gamma f}(x_k), \quad z_{k+1} = \operatorname{prox}_{\gamma g}(2 y_{k+1} - x_k), \quad x_{k+1} = x_k + z_{k+1} - y_{k+1}
```

**Parameters:**
- `gamma`: Step size parameter (defaulted automatically to `1.0` or `1 / L_f`).
- `maxit`: Maximum number of iterations.
- `tol`: Convergence tolerance.

**Pros:**
- ✅ Exact splitting for two proximable terms without requiring inner linear solves
- ✅ Direct support for hard consistency constraints and indicators

**Cons:**
- ❌ Restricted to at most two proximable terms

### Primal-Dual Hybrid Gradient (`ChambollePock`, also `PDHG`)

**When to use:**
- Regularizers composed with linear transforms — total variation, temporal TV, several at once —
  when a solve without inner CG iterations and without a penalty `ρ` to tune is wanted, on
  Cartesian data in particular.

**How it works:**
Chambolle-Pock's primal-dual method (Algorithm 1 of the 2011 paper) solves
``\min_x g(x) + h(Kx)`` with ``g`` and ``h`` used only through their proximal operators and ``K``
only through `K` and `K'`. `reconstruct` stacks every term into ``h``: the data term and each
regularizer become blocks of ``K = [\mathcal{A}; D_1; …]`` and ``h`` their separable sum, so the data
term is handled through its proximal operator, not its gradient. Each iteration applies
``\mathcal{A}``, ``\mathcal{A}'`` and every ``D_i``, ``D_i'`` once.

Unless `tau`, `sigma`, `ratio` or `normL` is given, `reconstruct` preconditions the iteration
(block-diagonal preconditioning, Pock & Chambolle 2011): the data block gets a dual step of its
own, per k-space sample proportional to the density-compensation weight on non-Cartesian data
(the acquisition's `dcf`, or `density_compensation`'s when it has none), and the step budget is
split evenly between the data and the regularization blocks. Without it one scalar step serves
every block, the large-norm data block dictates it, and the data dual barely moves.

**Parameters:**
- `tau`, `sigma`, `ratio`, `normL`: step sizes, the ratio ``\sigma/\tau`` and ``\|K\|`` of the
  unpreconditioned iteration; giving any of them turns the preconditioning off.
- `maxit`, `tol`.

**Pros:**
- ✅ No inner solve, no penalty parameter, no Lipschitz constant of the data term
- ✅ Any number of regularizers
- ✅ On the 2D 8-coil cases with anisotropic TV, NRMSE after 25 / 50 / 200 iterations: radial
  0.028 / 0.0165 / 0.0163 against `VuCondat`'s 0.32 / 0.23 / 0.070; Cartesian 0.083 / 0.072 /
  0.067 against 0.40 / 0.36 / 0.150

**Cons:**
- ❌ On non-Cartesian data an iteration costs about three `VuCondat` iterations: the data block
  applies the NFFT and its adjoint, not the Toeplitz normal operator. It still reaches a given
  accuracy several times sooner.
- ❌ Needs more iterations than ADMM; count operator applications, not iterations, when comparing
  the two

```julia
img = reconstruct(acq, IterativeReconstruction(TotalVariation2D(1e-2); algorithm = PDHG(), maxit = 500))
```

### Vũ-Condat (`VuCondat`)

**When to use:**
- The problems `ChambollePock` takes, with the data term handled through its gradient (the fused
  normal operator): one ``\mathcal{A}'\mathcal{A}`` per iteration, but a primal step capped by
  ``\|\mathcal{A}\|^2``, so it needs many more iterations than the preconditioned `ChambollePock`.

**How it works:**
The Vũ-Condat generalization of Chambolle-Pock solves ``\min_x f(x) + g(x) + h(Dx)``, taking the
smooth ``f`` (the data term, through its fused normal operator) by a gradient step and ``h`` through
its proximal operator. `reconstruct` supplies the gradient's Lipschitz constant ``\|\mathcal{A}\|^2``
as `beta_f`, which caps the primal step at about ``2/\|\mathcal{A}\|^2``.

**Parameters:**
- `gamma1`, `gamma2`: primal and dual step sizes, derived from `beta_f` and ``\|D\|`` when not
  given.
- `maxit`, `tol`.

```julia
img = reconstruct(acq, IterativeReconstruction(TotalVariation2D(1e-2); algorithm = VuCondat(), maxit = 500))
```

### Nonlinear Conjugate Gradient and L-BFGS (`NCG`, `LBFGS`)

**When to use:**
- Every term smooth but not every term quadratic: an edge-preserving (Huber) roughness penalty,
  a Tikhonov term next to it. `LBFGS` is the default for such a problem.

**How it works:**
Both minimize ``\sum_i f_i(L_i x)`` with a line search along each search direction ``d`` that keeps
``L_i x`` and ``L_i d`` and so applies no operator: an iteration costs one ``L_i`` and one ``L_i'`` per
term, and the data term enters through its normal operator, one ``\mathcal{A}'\mathcal{A}`` per
iteration. No Lipschitz constant is needed, so a Huber penalty with a small threshold ``\delta``,
which caps POGM's fixed step at about ``\delta/(8\lambda)``, does not slow them down. `NCG` is
Polak-Ribière+, `LBFGS` keeps the last `memory` correction pairs.

**Parameters:**
- `memory` (`LBFGS` only): stored correction pairs (default `5`).
- `eta`: line search accuracy, the accepted ``|\varphi'(\alpha)|`` relative to ``|\varphi'(0)|``
  (`0.1` for `NCG`, `0.9` for `LBFGS`).
- `maxit`, `tol` (on the size of the last step).

**Pros:**
- ✅ On a 2D 8-coil Cartesian case with ``\delta = 0.01``, 25 iterations reach NRMSE 0.044 where
  POGM reaches 0.081; with ``\delta = 0.001``, 0.19 against 0.34
- ✅ No step size and no ``\|\mathcal{A}\|`` estimate

**Cons:**
- ❌ Smooth terms only: a non-smooth regularizer or a constraint needs a proximal method

```julia
img = reconstruct(acq, IterativeReconstruction(EdgePreservingRoughness2D(1e-2; δ = 0.01); algorithm = LBFGS(), maxit = 50))
```

## Tuning Algorithm Parameters

### Maximum Iterations

**How many iterations do you need?**

Typical ranges:
- CG/CGNR: 10-50 iterations
- FISTA: 50-200 iterations
- ADMM: 20-100 iterations

**Strategy:**
```julia
# Start with more iterations to see convergence behavior
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = FISTA(), maxit = 200); verbosity = Verbose())
# Check output to see when convergence plateaus

# Then use fewer iterations in production
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = FISTA(), maxit = 80))
```

`maxit` and `reltol` on `IterativeReconstruction` take precedence over `maxit` and `tol` on the
algorithm object. To let the algorithm's own values through instead, set the method's to
`nothing`:

```julia
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = FISTA(maxit = 200), maxit = nothing))
```

### Convergence Tolerance

Controls early stopping:

```julia
# Stricter convergence
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = FISTA(), maxit = 200, reltol = 1e-6))

# Looser convergence (faster but less accurate)
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = FISTA(), maxit = 200, reltol = 1e-3))

# Disable early stopping
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = FISTA(), maxit = 100, reltol = 0))
```

Ristretto's tolerance is **relative**, which is why it is called `reltol`: the absolute threshold handed
to the solver is `max(10*eps, reltol * maximum(abs, x₀))`. `ProximalAlgorithms`' own `tol`, on the
algorithm object, is absolute.

**Practical tip:** Default `reltol=1e-4` is usually good. Tighten to 1e-5 or 1e-6 if you need higher accuracy.

### Verbosity and Monitoring

Track convergence:

```julia
# Show progress every iteration
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = algorithm); verbosity = Verbose(; freq = 1))

# Show progress every 10 iterations
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = algorithm); verbosity = Verbose(; freq = 10))

# A progress bar instead of the log
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = algorithm); verbosity = ProgressBar())

# No output — the default, so this is what a plain `reconstruct` call does
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = algorithm); verbosity = Silent())
```

**What to look for:**
- Objective value decreasing
- Changes becoming smaller
- Reasonable convergence rate

### Convergence Curves: `on_iteration` and `IterationTrace`

`verbosity` prints the iterations; `on_iteration` hands them to you. `IterativeReconstruction`
takes a callback that the solver invokes once per iteration with a single `NamedTuple`, so a
single `reconstruct` call is enough to plot error against iteration *and* against wall-clock time:

```julia
using Ristretto

nrmse(x, ref) = sqrt(sum(abs2, x .- ref) / sum(abs2, ref))

trace = IterationTrace(x -> nrmse(x, reference))
img = reconstruct(
    acq,
    IterativeReconstruction(L1Wavelet2D(0.01); algorithm = FISTA(), maxit = 60, on_iteration = trace);
    verbosity = Silent(),
)

using Plots
plot(trace.iterations, trace.values; xlabel = "iteration", ylabel = "NRMSE", yscale = :log10)
plot(trace.times, trace.values; xlabel = "wall-clock time (s)", ylabel = "NRMSE", yscale = :log10)
```

`IterationTrace` collects four columns: `iterations`, `times` (seconds since the solve started, on
a monotonic clock), `values` (the reduction applied to each iterate — the whole image if you leave
the reduction out), and `metrics` (the algorithm's own numbers). The clock starts *after* the
encoding operator and the operator-norm estimate are built, so `times` measures the solve, not the
setup — which is what you want when comparing two algorithms on the same problem.

The callback payload always carries `iteration`, `x` and `elapsed_ns`, plus `slice` when the
reconstruction was split into tasks. `x` is the current estimate already inverse-scaled and
carrying its dimension names — the same units, shape and type as the value `reconstruct`
returns, including a `ReconImage` holding the components on the `Component` path (without the acquisition's header). The remaining fields depend on
what the algorithm computes, and are *absent* rather than `nothing` when it computes nothing of
the sort:

| algorithm | additional fields |
|---|---|
| `FISTA`, `ISTA`, `POGM` | `objective`, `smooth_value`, `nonsmooth_value`, `stepsize`, `fixed_point_residual` |
| `DouglasRachford` | `objective`, `smooth_value`, `nonsmooth_value`, `fixed_point_residual` |
| `ADMM` | `primal_residual`, `dual_residual`, `iterate_change` |
| `CG`, `CGNR` | `residual_norm` |
| `ChambollePock` | `primal_change`, `dual_change` |
| `NCG`, `LBFGS` | `objective`, `stepsize`, `gradient_norm` |

So an objective-vs-iteration curve for FISTA is `[m.objective for m in trace.metrics]`, while for
ADMM the comparable curve is `[m.primal_residual for m in trace.metrics]`. Test
`haskey(info, :objective)` in a callback that has to cope with either.

A plain function works just as well when you only need a side effect:

```julia
reconstruct(acq, IterativeReconstruction(reg; on_iteration = info -> @info "iteration" info.iteration))
```

Two things to know:

- **Task splitting.** With several slabs in flight the callback fires from several tasks at once,
  and each payload gains a `slice` field naming its slab. `IterationTrace` appends under a lock,
  so it is safe as is — but group its columns by `trace.slices` before plotting, since the entries
  from different slabs interleave. A hand-written callback must be thread-safe itself.
- **Cost.** The callback copies the iterate every iteration, so a trace is not free — an NRMSE
  trace roughly doubles the solve time on a small problem. Not passing one costs nothing at all:
  with `on_iteration === nothing` no hook is installed in the solver loop, and a 128×128 solve
  allocates identically with and without the feature present.

`on_iteration` belongs to `IterativeReconstruction`, not to `ReconstructionConfig` — a direct
reconstruction has no iterations to observe. Passing it to `reconstruct` raises an error saying so.

```@docs
IterationTrace
```

## Advanced Usage

### Auto-Selecting Multiple Algorithms

Try multiple algorithms automatically:

```julia
# Provide tuple of candidate algorithms to try
algorithms = (CG(maxit=20), FISTA(maxit=100), ADMM(maxit=50))
img = reconstruct(acq, IterativeReconstruction(reg; algorithm = algorithms))

# Package automatically selects best for problem
# - CG tried first for smooth problems
# - FISTA for single regularizer
# - ADMM for multiple regularizers
```

### Custom Stopping Criteria

```julia
using Ristretto.ProximalAlgorithms

# Custom stopping function
function my_stop(iter, state)
    # Stop if objective doesn't change much
    if iter > 1
        rel_change = abs(state.objective - prev_obj) / abs(prev_obj)
        return rel_change < 1e-5
    end
    prev_obj = state.objective
    return false
end

# Use with low-level interface
# (requires working directly with ProximalAlgorithms.jl)
```

### Warm Starting

Use previous solution as initialization:

```julia
# First reconstruction
img1 = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(5e-3)))

# Use as initialization for refined reconstruction
img2 = reconstruct(acq, IterativeReconstruction(L1Wavelet2D(3e-3)); x₀=img1)
```

**When useful:**
- Parameter sweeps
- Iterative refinement
- Multi-stage reconstruction

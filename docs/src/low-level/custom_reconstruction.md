# Custom Reconstruction with StructuredOptimization

*Tutorial: [Low-level interface](../tutorials/12_low_level_interface.md).*

!!! note "Bundled package"
    Ristretto ships its own copy of StructuredOptimization.jl and the packages it builds on
    (AbstractOperators.jl, ProximalOperators.jl, ProximalAlgorithms.jl). Import them through
    Ristretto, as `using Ristretto.StructuredOptimization` and `using
    Ristretto.AbstractOperators`, and do not `Pkg.add` the registered packages: the bundled
    version carries work that is not registered yet (multithreading, GPU support, new operators,
    functions and algorithms). Upstreaming it is under way, and the bundled copy goes away once
    the registered releases have it. Documentation of the bundled version:
    [StructuredOptimization.jl (fork)](https://hakkelt.github.io/StructuredOptimization.jl/dev/),
    [AbstractOperators.jl (fork)](https://hakkelt.github.io/AbstractOperators.jl/dev/),
    [ProximalOperators.jl (fork)](https://hakkelt.github.io/ProximalOperators.jl/dev/),
    [ProximalAlgorithms.jl (fork)](https://hakkelt.github.io/ProximalAlgorithms.jl/dev/).

This page shows how to experiment with custom reconstruction problems using `StructuredOptimization.jl`, leveraging its convenient bindings to `AbstractOperators.jl` (operators like FFT, Wavelets, reshape, slicing) and `ProximalOperators.jl` (norms and penalties with fast proximal maps).

## Essentials

- `Variable`: declares decision variables (scalars, vectors, matrices, tensors).
- `@minimize`: builds and solves an optimization problem from expressions.
- `problem(...)` and `StructuredOptimization.parse_problem(...)`: build and inspect the parsed problem for a given algorithm.
- Convenient bindings:
  - Smooth terms: `ls(ex)` for 1/2||ex||^2
  - Nonsmooth norms: `norm(ex, 1)`, `norm(ex, 2)`, `norm(ex, Inf)`, mixed `norm(ex, 2, 1)`, nuclear norm `norm(ex, *)`
  - Operator composition: `op * expr`, `reshape(expr, ...)`, `fft`, `dct`, `finitediff`, etc.

```@setup so
using Random
Random.seed!(0)
```

## Building the Model Yourself

`reconstruct` assembles the optimization problem through `build_model`. The variant that also
hands back the variables is useful when a regularization introduces auxiliary variables of its
own — [`TotalGeneralizedVariation2D`](@ref) is the example — because the solver's variable
ordering is then no longer a reliable way to find the image.

```@docs
build_model
Ristretto.build_model_with_variables
Ristretto.materialize
Ristretto.materialize_with_auxiliaries
```

## Two-Variable Example: Sparse + Low-Rank Wavelet Prior

We build a toy reconstruction with two variables and two regularizers:
- Variable `x`: L1 sparsity prior.
- Variable `z`: nuclear norm prior after a wavelet transform and reshaping to a matrix.
- Data fidelity: simple least-squares to synthetic observation `b`.

```@example so
using Ristretto.StructuredOptimization
using Ristretto.AbstractOperators
using Ristretto.ProximalOperators
using Ristretto.WaveletOperators: WaveletOp, WT, wavelet
using SparseArrays: sprandn
using Ristretto.ProximalAlgorithms: FastForwardBackward, PANOCplus

# Problem size (kept small for docs)
nx, ny = 32, 32

# Variables
x = Variable(nx, ny)           # sparse image component
z = Variable(nx, ny)           # low-rank (after transform) component

# Synthetic observation b = x* + z* + small noise (unknown in practice)
x_true = sprandn(nx, ny, 0.1)
z_true = randn(nx, ny) .* 0.1
b = x_true + z_true + 0.01 .* randn(nx, ny)

# Wavelet transform operator (orthonormal, tight frame)
W = WaveletOp(Float64, wavelet(WT.db4), (nx, ny))

# Regularization strengths
λ1 = 0.05
λ2 = 0.2

# Build problem:
#   min_{x,z} 1/2 || (x + z) - b ||^2 + λ2 * nuclearnorm( W*z ) s.t. norm(x, 0) < 50
# It is hard to solve, so first we solve a relaxed version without the L0 constraint:
#   min_{x,z} 1/2 || (x + z) - b ||^2 + λ1 * norm(x, 1) + λ2 * nuclearnorm( W*z )
# Bindings used:
#   - ls(...)       -> least-squares
#   - norm(.,0)     -> L0 "norm" (count of nonzeros)
#   - norm(., 1)    -> L1 norm
#   - norm(., *)    -> nuclear norm (sum of singular values)
#   - W * z         -> operator application

# Parse first (inspect what a solver expects) -- optional, only for demonstration
p = problem( ls(x + z - b), λ2 * norm(W * z, *), norm(x, 0) <= 50 )
alg, kwargs, vars = StructuredOptimization.parse_problem(p, PANOCplus())
println("Prepared keys for PANOCplus: ", keys(kwargs))

# Solve relaxed problem with FISTA
(x̂, ẑ), it = @minimize ls(x + z - b) + λ1 * norm(x, 1) + λ2 * norm(W * z, *) with FastForwardBackward(maxit=50, verbose=false)
println("FISTA Iterations (relaxed problem): ", it)
# Solve original problem with PANOCplus
(x̂, ẑ), it = @minimize ls(x + z - b) + λ2 * norm(W * z, *) st norm(x, 0) <= 50 with PANOCplus(maxit=20, tol=1e-6, verbose=false)
println("Iterations: ", it)
println("Solution sizes: ", size(~x̂), ", ", size(~ẑ))

# Quick sanity: objective components (not rigorous tests)
val_l1 = NormL0(λ1)(~x̂)
# For nuclear norm value evaluate explicitly on the reshaped transform
using LinearAlgebra
S = svdvals(W * ~ẑ)
val_nuc = λ2 * sum(S)
println("L1 term: ", round(val_l1, digits=4), "; Nuclear term: ", round(val_nuc, digits=4))
```

### What Just Happened
- `ls(x + z - b)` is the data fidelity term.
- `norm(x, 1)` applies the L1 norm to `x` via `ProximalOperators.NormL1`.
- `norm(., 0)` applies the L0 "norm" (count of nonzeros) via `ProximalOperators.NormL0`.
- `norm(W*z, *)` applies the nuclear norm through a `Term(NuclearNorm(), ...)` binding; the reshape ensures a matrix domain.
- `W` is an orthonormal/Parseval wavelet transform (tight frame), enabling efficient proximal splitting.
- The problem separates across variables (`x` and `z`), so FISTA (= FastForwardBackward) can handle both nonsmooth terms.


## Anatomy: Basics in One Place

```@example so
using Ristretto.StructuredOptimization
using Ristretto.AbstractOperators
using Ristretto.ProximalOperators
using Ristretto.ProximalAlgorithms: FastForwardBackward

# Variables and access
u = Variable(10); v = Variable(10)

# Smooth term (LS)
term_smooth = ls(u + v - randn(10))

# Common nonsmooths
term_l1  = norm(u, 1)        # L1
term_l2  = norm(v, 2)        # L2
term_l21 = norm(reshape(v, 2, 5), 2, 1)  # group sparsity
term_nuc = norm(reshape(v, 5, 2), *)     # nuclear

# AbstractOperators bindings
ex_fft  = fft(u)   # Fourier transform binding
ex_resh = reshape(ex_fft, 5, 2)                   # reshape expression

# Build and solve explicitly via `problem` + `solve`
q = problem(term_smooth, 0.1*term_l1, 0.05*term_nuc)
sol, it2 = solve(q, FastForwardBackward(maxit=50, verbose=false))
println("Solved in ", it2, " iterations. Size(~u): ", size(~u))
```

## Design of the Method API

`reconstruct(acq, method)` takes one method object that carries everything that changes the
problem: the kind of reconstruction, its regularization, its solver and its iteration control
(`maxit`, `reltol`, `algorithm` are rejected as `reconstruct` keywords). Run-level settings that
leave the problem unchanged — scaling, verbosity, threading, task executor — stay keywords of
`reconstruct`, collected in [`ReconstructionConfig`](@ref). One method
object, rather than positional regularization and algorithm arguments, is what lets a GRAPPA, a
partial-Fourier method and an unregularized CG solve share the one entry point.

- **Two abstract kinds.** `ReconstructionMethod` splits into `IterativeMethod`, whose concrete type
  is [`IterativeReconstruction`](@ref) (an objective handed to a proximal or gradient solver), and
  `DirectMethod`, a closed-form or fixed-point algorithm with its own loop:
  [`DirectReconstruction`](@ref), [`GRAPPA`](@ref), [`SPIRiT`](@ref), [`Homodyne`](@ref),
  [`POCS`](@ref), [`PhaseConstrained`](@ref). Named methods are types, not functions returning a
  configured `IterativeReconstruction`, so they dispatch, print and fail under their own name.
- **Lowering.** `reconstruct` first calls `lower(method, acq)`, which may rewrite a method into the
  one that executes it: `DirectReconstruction()` resolves its coil combination against the
  acquisition, and `SPIRiT(; iterative = true)` becomes an `IterativeReconstruction` with a
  calibrated [`SPIRiTConsistency`](@ref) term, so the iterative path exists once.
- **Configuration axes are types.** Every choice that selects code — data fidelity
  ([`L2Loss`](@ref), [`HardConsistency`](@ref), [`NoFidelity`](@ref)), coil combination, signal
  model, partial-Fourier filter, scaling, verbosity — is a type and a type parameter of its owner,
  never a `Symbol`, so it is resolved by dispatch and can be extended downstream.
- **What the variable is, is a signal model.** `signal_model = nothing` optimizes the image;
  [`TemporalBasis`](@ref) optimizes subspace coefficients through $\mathcal{A}\Phi$;
  [`KSpaceToImage`](@ref) optimizes the multi-channel k-space with $\mathcal{P}$ as the encoding
  operator, followed by an inverse FFT and a coil combination. There is no separate domain axis:
  every regularizer acts on the variable as it is, which is why the k-space priors
  ([`StructuredLowRank`](@ref), [`SPIRiTConsistency`](@ref)) are ordinary regularizers paired
  with `KSpaceToImage`, and no inverse Fourier transform is inserted for an image-domain prior.
- **Applicability is checked up front.** `check_applicable(method, acq)` runs after lowering and
  before any operator is built, so a mismatch is an `ArgumentError` naming the requirement rather
  than a shape error deep in a solve: [`GRAPPA`](@ref) rejects non-Cartesian data and irregular
  phase-encoding patterns, and an explicit `AdjointSensitivity()` rejects an acquisition without
  sensitivity maps. It is the one hook a new method overrides (see
  [Reconstruction Methods](../high-level/methods.md)).
- **Task splitting follows the method.** Every regularizer and the signal model report the
  dimensions they couple through `get_affected_dims`; the remaining batch dimensions are solved as
  independent tasks (see [Task Splitting](../high-level/task_splitting.md)). `TemporalBasis`
  couples the time dimension and `KSpaceToImage` couples every dimension, so a k-space solve is
  never split. A direct method couples nothing beyond its encoding dimensions (and the coil axis
  it combines), so every other batch dimension is eligible.

## Where to Go Next

- More norms and penalties: see `low-level/proximal_operators.md`.
- Operator catalog and composition patterns: see `low-level/abstract_operators.md` and `low-level/operators.md`.
- For full MRI forward models and reconstruction entry points, see `high-level/reconstruction.md`.

## See Also

- [ProximalOperators.jl Summary](@ref): list of common proximal functions and custom implementation pattern.
- [AbstractOperators.jl Summary](@ref abstract_operators): base operator abstractions and composition rules.
- [MRI Operators](@ref): concrete Fourier, sensitivity map, and subsampling operators.
- [High level Reconstruction](@ref reconstruction): unified reconstruction interface tying everything together.

# Custom Reconstruction with StructuredOptimization

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

`StructuredOptimization.jl` states an optimization problem the way it is written on paper, over
`AbstractOperators` and `ProximalOperators`, and picks a solver for it. `reconstruct` builds its
problems this way; [the low-level interface](../tutorials/12_low_level_interface.md) writes one
by hand (§3–4).

## Essentials

- `Variable`: declares decision variables (scalars, vectors, matrices, tensors).
- `@minimize`: builds and solves an optimization problem from expressions.
- `problem(...)` and `StructuredOptimization.parse_problem(...)`: build and inspect the parsed problem for a given algorithm.
- Convenient bindings:
  - Smooth terms: `ls(ex)` for 1/2||ex||^2
  - Nonsmooth norms: `norm(ex, 1)`, `norm(ex, 2)`, `norm(ex, Inf)`, mixed `norm(ex, 2, 1)`, nuclear norm `norm(ex, *)`
  - Operator composition: `op * expr`, `reshape(expr, ...)`, `fft`, `dct`, `finitediff`, etc.

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

## See Also

- [ProximalOperators.jl Summary](@ref): the common proximal functions.
- [AbstractOperators.jl Summary](@ref abstract_operators): base operator abstractions and composition rules.
- [MRI Operators](@ref): concrete Fourier, sensitivity map, and subsampling operators.
- [High level Reconstruction](@ref reconstruction): unified reconstruction interface tying everything together.

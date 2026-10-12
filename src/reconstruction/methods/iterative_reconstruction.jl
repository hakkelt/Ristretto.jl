"""
	IterativeReconstruction{R, A, F<:DataFidelity, M, C} <: IterativeMethod

Configures an iterative reconstruction problem with regularization terms, solver algorithms,
data fidelity, and signal modeling options.

# Fields
- `regularization::R`: Tuple of regularization terms (`Regularization` or `Component`).
- `algorithm::A`: Solver algorithm or tuple of candidate algorithms.
- `fidelity::F`: Data fidelity term (default `L2Loss()`).
- `signal_model::M`: Signal model mapping the optimization variable to the image (default `nothing`);
  e.g. `TemporalBasis` for subspace reconstruction or `KSpaceToImage` for a k-space-domain solve.
- `exact_opnorm::Bool`: Compute `‖𝒜‖` with a fully converged iteration (`opnorm`) rather than with
  `AbstractOperators.estimate_opnorm` (default `false`). That iteration converges from below
  and stops short, so the result is a slight *under*-estimate and `Lf` comes out slightly too
  small — the unsafe direction, since no backtracking runs to correct it. The default estimate
  errs upwards instead. Use this when the number itself is wanted, not to make a step size safer,
  and in particular not with POGM, which has been observed to diverge on a 1.2% low `Lf`.
- `disable_operator_normalization::Union{Nothing, Bool}`: Skip the `‖𝒜‖` estimate and let the
  algorithm derive its own step size (default `nothing` for auto-detection: skipped for pure
  unregularized CG/CGNR, which is scale invariant, run for proximal algorithms). The name predates
  the change that stopped `‖𝒜‖` being used to rescale `𝒜`; it is a step-size switch only.

`𝒜` is left at its natural norm and `Lf = n‖𝒜‖²` is passed to the algorithm instead, so the problem
solved is `½‖𝒜x - y‖² + R(x)`: `λ` weights the regularizer in the data's own units and the result
comes back in them. See "Operator norm, step size and λ" in `docs/src/high-level/methods.md` for
what changed and how to migrate a `λ` tuned against the previous behaviour.
- `maxit::Union{Nothing, Int}`: Maximum solver iterations (default `100`). `nothing` defers to the
  `algorithm`'s own `maxit`, which is how an `algorithm = FISTA(maxit = 500)` is honoured.
- `reltol::Union{Nothing, Float64}`: Stopping tolerance (default `1e-4`), **relative** — hence the
  name: the absolute threshold handed to the solver is `max(10*eps, reltol * maximum(abs, x₀))`,
  where `x₀` is the initial guess. `ProximalAlgorithms`' own `tol` is absolute, which is why this
  one is not called `tol`. `0` disables the tolerance test and `nothing` defers to the
  `algorithm`'s own stopping criterion.

- `on_iteration::C`: `nothing` (default) or a callback invoked once per solver iteration; see
  "Observing the iterations" below.

`maxit` and `reltol` are keyword-only on every constructor; regularization terms are the only
positional arguments.

# Observing the iterations

`on_iteration = f` makes the solver call `f(info)` once per iteration, with a single `NamedTuple`
carrying at least

- `iteration::Int` — 1-based count of completed iterations,
- `x` — the current image estimate, already inverse-scaled and re-wrapped as a `NamedDimsArray`
  (or a `ReconImage` holding the components on the `Component` path), i.e. in the same units and shape as the value
  `reconstruct` will return,
- `elapsed_ns::UInt64` — nanoseconds since the solve started, from the monotonic `time_ns` clock,
- `slice::String` — present only when the reconstruction was split into tasks, naming the slab.

Algorithm-dependent fields are present only where the algorithm actually computes them, and are
*absent* rather than `nothing` when it does not:

| algorithm | extra fields |
|---|---|
| `POGM` / `FISTA` / `ISTA` (forward-backward) | `objective`, `smooth_value`, `nonsmooth_value`, `stepsize`, `fixed_point_residual` |
| `DouglasRachford` | `objective`, `smooth_value`, `nonsmooth_value`, `fixed_point_residual` |
| `ADMM` | `primal_residual`, `dual_residual`, `iterate_change` |
| `CG` / `CGNR` | `residual_norm` |
| `ChambollePock` | `primal_change`, `dual_change` (`‖x - x⁻‖∞`, `‖y - y⁻‖∞`) |
| `NCG` / `LBFGS` | `objective`, `stepsize`, `gradient_norm` |

Use [`IterationTrace`](@ref) rather than writing a collector by hand. The callback fires from the
solver task, so under task splitting several slabs may call it concurrently — `IterationTrace`
takes a lock; a hand-written callback must be thread-safe itself.

When `on_iteration === nothing` nothing is installed in the solver loop at all: the hook is
dispatched away at compile time, so an unobserved reconstruction pays neither a branch nor an
allocation for this feature.

```julia
trace = IterationTrace(x -> nrmse(x, reference))
reconstruct(acq, IterativeReconstruction(L1Wavelet2D(0.01); maxit = 60, on_iteration = trace))
trace.iterations, trace.times, trace.values  # for an NRMSE-vs-iteration / -vs-time plot
```
"""
struct IterativeReconstruction{R <: Tuple, A, F <: DataFidelity, M, C} <: IterativeMethod
    regularization::R
    algorithm::A
    fidelity::F
    signal_model::M
    exact_opnorm::Bool
    disable_operator_normalization::Union{Nothing, Bool}
    maxit::Union{Nothing, Int}
    reltol::Union{Nothing, Float64}
    on_iteration::C

    function IterativeReconstruction(
            regularization::Tuple,
            algorithm,
            fidelity::F,
            signal_model::M,
            exact_opnorm::Bool,
            disable_operator_normalization::Union{Nothing, Bool};
            maxit::Union{Nothing, Integer} = 100,
            reltol::Union{Nothing, Real} = 1.0e-4,
            on_iteration::C = nothing,
        ) where {F <: DataFidelity, M, C}
        _validate_regularization(regularization)
        return new{typeof(regularization), typeof(algorithm), F, M, C}(
            regularization,
            algorithm,
            fidelity,
            signal_model,
            exact_opnorm,
            disable_operator_normalization,
            isnothing(maxit) ? nothing : Int(maxit),
            isnothing(reltol) ? nothing : Float64(reltol),
            on_iteration,
        )
    end
end

function _validate_regularization(regs::Tuple)
    has_comp = any(r -> r isa Component, regs)
    has_bare = any(r -> r isa Regularization, regs)
    if has_comp && has_bare
        throw(ArgumentError("Cannot mix bare regularization terms with `Component`s; wrap loose regularization terms in a `Component`."))
    end
    return regs
end

# Keyword constructor
function IterativeReconstruction(;
        regularization = (),
        algorithm = DEFAULT_ALGORITHMS,
        fidelity::DataFidelity = L2Loss(),
        signal_model = nothing,
        exact_opnorm::Bool = false,
        disable_operator_normalization::Union{Nothing, Bool} = nothing,
        maxit::Union{Nothing, Integer} = 100,
        reltol::Union{Nothing, Real} = 1.0e-4,
        on_iteration = nothing,
    )
    regs_tuple = ensure_tuple(regularization)
    return IterativeReconstruction(
        regs_tuple,
        algorithm,
        fidelity,
        signal_model,
        exact_opnorm,
        disable_operator_normalization;
        maxit,
        reltol,
        on_iteration,
    )
end

# Positional vararg constructor (requires at least 1 term to prevent collision with keyword constructor)
function IterativeReconstruction(
        reg::Union{Regularization, Component},
        more_regs::Union{Regularization, Component}...;
        algorithm = DEFAULT_ALGORITHMS,
        fidelity::DataFidelity = L2Loss(),
        signal_model = nothing,
        exact_opnorm::Bool = false,
        disable_operator_normalization::Union{Nothing, Bool} = nothing,
        maxit::Union{Nothing, Integer} = 100,
        reltol::Union{Nothing, Real} = 1.0e-4,
        on_iteration = nothing,
    )
    regs = (reg, more_regs...)
    return IterativeReconstruction(;
        regularization = regs,
        algorithm,
        fidelity,
        signal_model,
        exact_opnorm,
        disable_operator_normalization,
        maxit,
        reltol,
        on_iteration,
    )
end

# Rebuild `method` with a different regularization tuple, every other field carried over. The
# task-splitting path needs this per slice (λ is scale-compensated there); going through the
# keyword constructor keeps that call site from having to be edited every time a field is added.
function _with_regularization(method::IterativeReconstruction, regs::Tuple)
    return IterativeReconstruction(;
        regularization = regs,
        algorithm = method.algorithm,
        fidelity = method.fidelity,
        signal_model = method.signal_model,
        exact_opnorm = method.exact_opnorm,
        disable_operator_normalization = method.disable_operator_normalization,
        maxit = method.maxit,
        reltol = method.reltol,
        on_iteration = method.on_iteration,
    )
end

progress_total(method::IterativeReconstruction, acq_data) = method.maxit

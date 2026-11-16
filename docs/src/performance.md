# Performance

For end users, the main performance recommendations are:

1. Prefer `mul!` over `*` to avoid allocations.
2. Preallocate with `allocate_in_domain(op)` / `allocate_in_codomain(op)`.
3. Use operator constructors with array inputs to preserve storage type (CPU vs GPU).

## Accumulating multiplication

`mul!(y, A, x, α, β)` computes `y = α * (A * x) + β * y`, as for matrices. Use it to add an
operator's output to an array, or to scale it, without a separate pass over `y`:

```julia
mul!(y, A, x, 1, 1)      # y += A * x
mul!(y, A', z, -τ, 1)    # y -= τ * A' * z
mul!(y, A, x, 2, false)  # y = 2 * A * x; the old contents of y are not read
```

`Eye`, `Zeros`, `DiagOp`, `MatrixOp`, `LMatrixOp`, `FiniteDiff`, `HigherOrderDiff`, `GetIndex`
and `ZeroPad`, and their adjoints, compute it in one pass over `y`; `Scale` and `Reshape` pass
`α` and `β` on to the operator they wrap. Any other operator computes `A * x` first and then
combines it with `y`. With `β` zero that needs no memory beyond `y`. Otherwise it needs an array
the size of the codomain, which inside [`with_operator_pool`](@ref) comes from the pool and goes
back to it after the call, and outside a pool is allocated for the call:

```julia
pool = OperatorPool()
with_operator_pool(pool) do
    for k in 1:100
        mul!(y, A, x, 1, 1)  # allocates on the first call only
    end
end
```

```@docs
mul!(::AbstractArray, ::AbstractOperator, ::Any, ::Number, ::Number)
```

Developer-oriented performance internals (threading heuristics, storage traits, backend caveats) were moved to [Custom Operators](@ref) to keep this page concise.

## Fused element-wise operators

Inside a `Compose`, consecutive element-wise operators (diagonals, masks, broadcasts and their
adjoints, a scaling, a DFT's normalization) run as one loop, which reads the run's input once and
writes its output once instead of passing every intermediate result through memory. The result is
the same bit for bit as applying the operators one by one. To measure what this is worth, switch
it off at run time:

```julia
AbstractOperators.POINTWISE_FUSION[] = false   # every operator runs its own mul!
AbstractOperators.POINTWISE_FUSION[] = true    # the default
```

```@docs
AbstractOperators.POINTWISE_FUSION
```

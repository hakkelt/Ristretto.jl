# Section: time-to-target-accuracy — the fair cross-toolkit comparison.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_accuracy_race.jl --threads=N [--use-mkl] [--data=synthetic|real|all] [--cases=...]
#
# ## Why not a fixed iteration count
#
# Every other section pins iterations and compares wall time. That is only meaningful if one
# iteration means the same work everywhere, and it does not:
#
#   * BART's ADMM `-i N` is a budget of ≈N normal-operator applications, not N outer iterations
#     (`src/iter/admm.c:453`), and its inner CG stops at `1e-3 · ‖rhs‖` with `cg_eps` hardcoded at
#     `src/iter/iter.c:106` — **not** settable from the CLI. With a warm start it often takes 0
#     inner iterations, so at ρ = 0.05 `-i 20` buys 18 outer iterations whose x-update never runs
#     and an NRMSE of 0.096, where Ristretto's 20 genuine iterations reach 0.003. Neither number is
#     wrong; they are answers to different questions.
#   * Even the TV rows that share a penalty (isotropic: Ristretto, BART, SigPy; anisotropic: Ristretto, BART,
#     SigPy, MRIReco) differ at the boundary (Ristretto mirrored, SigPy circular, MRIReco `GradientOp`
#     truncated), so no single λ is comparable and each toolkit needs its own.
#   * BART and MRIReco additionally pay one-off Lipschitz estimates on their FISTA paths (`-e` is
#     30 `𝒜ᴴ𝒜`; `power_iterations` is 2–30) that a per-iteration accounting hides.
#
# So: sweep the iteration count per toolkit at its own calibrated λ (and, for an ADMM row, its own
# calibrated penalty ρ, since each toolkit scales its operator differently), and report the wall time at
# the first count that reaches a common NRMSE target. That is the number a user actually cares
# about — "how long until the picture is this good" — and it is invariant to all of the above.
#
# The (case, method) pairs are `race_methods` of every catalog case (`_methods.jl`). The target is
# the case's `race_target` from `calibrate_lambda.jl` (the worst toolkit's best converged NRMSE ×
# 1.10, so every toolkit can reach it); the ADMM and PDHG rows of one TV share the larger of their
# two targets, so the two algorithms race to the same accuracy. A case that was never calibrated falls back to 1.10 × Ristretto's
# NRMSE at the top of its ladder, which is logged: that target is reachable by Ristretto by construction
# and says nothing about whether the others can reach it.
#
# BART's time is the `Total Time` `pics` reports itself (`time_bart`), without the process start and
# the cfl file I/O, so every figure is solver time.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
include(joinpath(@__DIR__, "_methods.jl"))

# Iteration ladders. BART's ADMM ladder is in `-i` budget units, hence much larger numbers; the
# heavy cases (3D, cine) climb a shorter ladder, since each rung is a full solve.
const LADDER = [3, 5, 8, 12, 20, 30, 50]
const LADDER_HEAVY = [5, 10, 20, 40]
const LADDER_BART_ADMM = [10, 20, 40, 80, 150, 300]
const LADDER_BART_ADMM_HEAVY = [20, 50, 100, 200, 400]

# A PDHG iteration applies the operator about once, where an ADMM iteration applies it once per
# inner CG step (see `PDHG_ITERATIONS`), so a PDHG row climbs the longer ladder of normal-operator
# budgets that BART's rows already use. Every toolkit gets it: on the 50-iteration ladder no PDHG
# row but MRpro's reached the target, while BART's reached it on this one.
ladder(c::BenchCase) = c.heavy ? LADDER_HEAVY : LADDER
long_ladder(c::BenchCase) = c.heavy ? LADDER_BART_ADMM_HEAVY : LADDER_BART_ADMM
ladder(c::BenchCase, method::Symbol) = haskey(PDHG_METHODS, method) ? long_ladder(c) : ladder(c)
bart_admm(m::Symbol) = m !== :wavelet

"""
    race(label, c, method, ladder, target, run) -> (iterations, time_ms, nrmse) or nothing

Walk `ladder`, stopping at the first entry whose `method` reconstruction of case `c` reaches `target`.
`run(it)` must return `(time_ms, image)`. Returns `nothing` when the target is never reached, which
is itself a result worth reporting — it means the toolkit cannot get there at this λ.
"""
function race(label, c::BenchCase, method::Symbol, ladder, target, run)
    for it in ladder
        t_ms, x = try
            run(it)
        catch e
            @warn "$label failed" it exception = (e, catch_backtrace())
            continue
        end
        err = mag_nrmse(_score_image(c, method, x), c.reference)
        @info @sprintf(
            "%-40s it=%4d  %9.1f ms  NRMSE=%.5f%s", label, it, t_ms, err,
            err <= target ? "  <= target" : ""
        )
        err <= target && return (it, t_ms, err)
    end
    @warn "$label never reached target" target
    return nothing
end

function addrow!(c::BenchCase, method::Symbol, target, fw, r)
    r === nothing && return nothing
    label = @sprintf("%s (NRMSE≤%.4g, %d it)", METHOD_LABEL[method], target, r[1])
    push!(results, BenchResult("Accuracy race", label, fw, NUM_THREADS, r[2], r[3], 0.0, c.id, c.source))
    return nothing
end

function ristretto_race_run(c::BenchCase, method::Symbol, λ, rho)
    return it -> begin
        t, _, x = time_run(ristretto_reconstructor(c, method; λ, rho, maxit = it, device = RISTRETTO_DEVICE); runs = timed_runs(c))
        (1000 * t, parent(x))
    end
end

for c in section_cases(_ -> true), method in race_methods(c)
    should_run(c.id, METHOD_LABEL[method]) || should_run("Accuracy race", METHOD_LABEL[method]) || continue
    λ_default = default_lambda(c, method)
    ristretto_rho = something(load_rho(c, method, "Ristretto"), admm_rho(c))
    ristretto_run = ristretto_race_run(c, method, load_lambda(c, method, "Ristretto", λ_default), ristretto_rho)
    target = load_race_target(c, method, NaN)
    if isnan(target)
        _, x = ristretto_run(last(ladder(c, method)))
        target = 1.1 * mag_nrmse(_score_image(c, method, x), c.reference)
        @warn "$(c.id) $method has no calibrated race target; using 1.10 × Ristretto's NRMSE at $(last(ladder(c, method))) it" target
    end
    println("--> $(c.id): $(METHOD_LABEL[method])  (target NRMSE ≤ $(round(target; sigdigits = 4)))")

    addrow!(c, method, target, FW, race("Ristretto $(c.id) $method", c, method, ladder(c, method), target, ristretto_run))

    for tk in COMPETITORS
        fw = framework_label(tk)
        (supports(tk, c, method) && should_run_framework(fw)) || continue
        λ = load_lambda(c, method, toolkit_key(tk), λ_default)
        ρ = load_rho(c, method, toolkit_key(tk))
        r = if tk === :bart && bart_admm(method)
            race(
                "$fw $(c.id) $method", c, method, long_ladder(c), target,
                it -> (RUNS[] = timed_runs(c); bart_run(c, method; λ, maxit = it, budget = it, ρ = something(ρ, CMP_RHO))),
            )
        else
            race(
                "$fw $(c.id) $method", c, method, ladder(c, method), target,
                it -> toolkit_run(tk, c, method; λ, ρ, maxit = it),
            )
        end
        addrow!(c, method, target, fw, r)
    end
    flush_results!("accuracy_race")
end

write_section("accuracy_race")

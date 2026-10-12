# Section: direct (adjoint) reconstruction of every Cartesian catalog case — Ristretto / SigPy / BART /
# MRIReco / MIRT / MRpro.
#   julia --project=benchmark/comparison -t N benchmark/comparison/scripts/run_base.jl --threads=N [--use-mkl] [--data=synthetic|real|all]
#
# Every toolkit returns Σ conj(Sᶜ) xᶜ, scored after dividing by Σ|Sᶜ|² (`_score_image`).
#
# **BART is not timed in this section — it cannot be, and a timed row would be noise.** The
# in-process adjoints here take milliseconds, while a `bart` invocation costs ~100–130 ms of process
# spawn plus disk I/O, with ±30 ms of run-to-run jitter on the login node (measured: `bart version`
# 94–124 ms, `bart copy` 152–215 ms, `bart fft -i 3` 163–253 ms). `fft` and `nufft` do not report
# their own run time as `pics` does (`time_bart`), and subtracting an estimate of the spawn and I/O
# from a 2 ms compute leaves a difference far inside that jitter, which once came out as 0.01 ms —
# 300× faster than everyone else, an artifact. BART still runs and its
# output is still checked against Ristretto's, but its `time_ms` is recorded as unmeasurable (-1). The
# iterative sections are unaffected: there the solver dominates the fixed overhead.
include(joinpath(@__DIR__, "_setup.jl"))
include(joinpath(@__DIR__, "_toolkits.jl"))
include(joinpath(@__DIR__, "_methods.jl"))

for c in section_cases(c -> c.trajectory === :cartesian)
    run_method_rows!("Base", c, :adjoint)
end

write_section("base")

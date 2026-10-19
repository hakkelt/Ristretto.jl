module FFTWOperators

using ..AbstractOperators
using FFTW, LinearAlgebra
using Base.Cartesian: @ncall
using Polyester: @batch
using FastBroadcast: @..
import LinearAlgebra: mul!
import Base: size, ndims

import ..AbstractOperators:
    _normalize_array_type,
    _array_wrapper_type,
    domain_type,
    codomain_type,
    fun_name,
    get_normal_op,
    has_optimized_normalop,
    allocate_in_domain,
    allocate_in_codomain,
    domain_array_type,
    codomain_array_type,
    can_be_combined,
    combine,
    _slice_operator,
    is_thread_safe,
    is_AcA_diagonal,
    is_AAc_diagonal,
    diag_AcA,
    diag_AAc,
    is_orthogonal,
    is_invertible,
    is_full_row_rank,
    is_full_column_rank,
    is_symmetric,
    has_fast_opnorm,
    check,
    is_threaded,
    supports_threading,
    _resolve_threaded,
    _elementwise_threaded,
    threading_threshold,
    _copy_operator_impl

"""
	fftw_threading_threshold(kind::Symbol) -> Int

Element count at which threading an FFTW transform of this `kind` starts to pay.

PROVENANCE: measured on 2026-09-24 (AMD EPYC 7763, one 16-core NUMA domain, `FFTW.ESTIMATE`,
OPENBLAS_NUM_THREADS=1), timing each operator's forward plus adjoint `mul!` planned with one
thread against the same operator planned with every thread, in a process started with
`-t 2`, `-t 4`, `-t 8` and `-t 16`, once with each Julia thread pinned to a core and once
without. Sizes are even powers of two, `Float32` and `Float64`, 1D, square 2D and (c2c) 2D
over 8 coils. The threshold is the smallest swept size from which threading is never slower
at *any* of those thread counts, pinned or not -- the loss below it is large, the gain at it
is not:

| kind | largest losing size | there (worst, 2-16 threads) | threshold | at the threshold |
|---|---|---|---|---|
| `:c2c` (DFT/IDFT) | 2^15 (64x64x8) | 0.35x-0.50x | 2^16 | 1.6x-6.7x (256x256 `Float32`) |
| `:r2r` (DCT/IDCT) | 2^14 (128x128) | 0.53x-0.73x | 2^16 | 1.05x-5.0x (256x256) |
| `:r2c` (RDFT/IRDFT) | 2^16 (256x256) | 0.63x-0.92x | 2^18 | 1.15x-6.0x |

The previous thresholds (c2c 2^13, the real transforms 2^14) came from an 8-thread process
on an EPYC 7352 with its threads unpinned, which is the kindest case for a threaded plan: a
128x128 `ComplexF32` transform, 16384 elements and above the old c2c threshold, measures 2.2x
there but 0.24x at 2 threads, and 0.23x-0.39x at every thread count once the threads are
pinned. A threaded plan below these sizes pays a fixed ~0.1-0.2 ms per call to wake and join
its helpers, which is several times the transform itself. Two cells past the thresholds still
lose, both `Float64` at 8-16 threads (256x256 0.2x-0.6x, 512x512 0.5x at 16), and both are
FFTW's `ESTIMATE` planner picking a poor threaded plan rather than a size effect: the sizes
around them win.

Below the thresholds threading an FFT is a *large* pessimisation, not a wash -- a 1024-element
c2c transform measures 0.01x-0.25x -- which is why the policy applies here rather than
trusting FFTW's planner to sort it out.
"""
fftw_threading_threshold(kind::Symbol) = kind === :r2c ? 2^18 : 2^16

"""
	_fftw_num_threads(kind, num_threads, threaded, n) -> Int

Resolve the plan-time FFTW thread count.

`num_threads` is FFTW's own vocabulary and is an explicit **command**: given, it wins
outright, which keeps an escape hatch for callers who know what they want. `threaded` is the
package-wide keyword and follows the package-wide rule -- `false` vetoes, `true`/`nothing`
enable subject to the policy above. See `AbstractOperators._resolve_threaded`.

Past the threshold the whole pool is used. An intermediate count was measured and rejected
(AMD EPYC 7352, 8 threads, `FFTW.ESTIMATE`, `ComplexF64` in-place c2c, min of 300 reps,
2026-09-20): sweeping every thread count over a grid of transform size x batch count, capping
to two threads in the first octave above the threshold and four in the second costs more than
it saves -- 1.47x on a 2^15 DCT and 1.64x on a 256x256 one at 2^16, against at most 1.45x
bought in a single c2c cell (64x64 x4 at 2^14: 25.52 us at four threads, 37.07 at eight).
The full count is the best or within 15 % of the best on every other cell measured, and never
slower than one thread anywhere above the threshold.

What `n` genuinely cannot express is **nesting**: an 8-thread plan applied from an
already-saturated outer loop oversubscribes. Measured at `nthreads()` concurrent tasks each
applying its own plan, an 8-thread plan is 7-32 % slower than a 1-thread one (64^2x8: 213.22
vs 162.50 us; 320^2x8: 11896.72 vs 11116.78), while at half that concurrency it is still
1.4-1.9x faster. The crossover is a property of the caller, not of the transform, so it
belongs to the caller: pass `threaded = false` or an explicit `num_threads` when planning an
operator that will be applied from inside a threaded region.
"""
function _fftw_num_threads(kind::Symbol, num_threads, threaded::Bool, n::Int)
    num_threads !== nothing && return Int(num_threads)
    use = _resolve_threaded(threaded) do
        Threads.nthreads() > 1 && n >= fftw_threading_threshold(kind)
    end
    return use ? Threads.nthreads() : 1
end

"""
	_with_fftw_threads(f, num_threads)

Run the planning callable `f` with FFTW's global thread count temporarily set to
`num_threads`, restoring the previous value afterwards.

FFTW's thread count is process-global state consulted at *plan* time, so it has to be set
around planning and put back; forgetting the restore would silently change the thread count
of every plan built later in the session.
"""
function _with_fftw_threads(f, num_threads::Int)
    prev = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    try
        return f()
    finally
        FFTW.set_num_threads(prev)
    end
end

include("DFT.jl")
include("RDFT.jl")
include("IRDFT.jl")
include("DCT.jl")
include("Shift.jl")
include("combination_rules.jl")

end # module FFTWOperators

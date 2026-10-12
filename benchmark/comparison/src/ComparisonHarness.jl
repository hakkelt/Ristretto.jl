module ComparisonHarness

using LinearAlgebra: norm
using Test: @test

include("bart_bridge.jl")
include("mirt_bridge.jl")

"""
Whether `MATLAB.jl` could be loaded, which it can only be where a MATLAB installation is visible.

Nothing measured here comes from MATLAB: the bridge exists for the optional cross-checks against
the authors' reference implementations (LORAKS 2.0, ESPIRiT, the primal-dual toolbox), and every
call site of `setup_matlab_paths` is currently commented out. `MATLAB.jl` cannot even be
precompiled where no MATLAB installation is visible, which took `Pkg.instantiate` and with it the
whole comparison suite down on a machine that has only BART, SigPy and MRIReco -- so it is not a
declared dependency of `benchmark/comparison` any more. Add it back
(`Pkg.add("MATLAB"); Pkg.build("MATLAB")` with a MATLAB module loaded) to enable the bridge; this
include then finds it and `MATLAB_AVAILABLE` becomes true.
"""
const MATLAB_AVAILABLE = try
    include("matlab_bridge.jl")
    true
catch err
    @warn "MATLAB is unavailable; the optional MATLAB reference cross-checks are disabled" err
    false
end
using .BARTBridge
using .MIRTBridge
if MATLAB_AVAILABLE
    @eval using .MATLABBridge
else
    setup_matlab_paths(; require = ()) = error(
        "MATLAB is not available in this environment, so the reference implementations under " *
            "benchmark/comparison/original_implementations cannot be called. Load a MATLAB module " *
            "(`module load matlab/...`) and rerun `Pkg.build(\"MATLAB\")` if they are needed."
    )
end

export run_bart, run_bart_timed
export setup_matlab_paths
export MIRT
export nrmse, check_nrmse

"""
    nrmse(x, xref)

Normalized root-mean-square error `‖x - xref‖ / ‖xref‖` (Frobenius over all entries).
"""
nrmse(x, xref) = norm(vec(x) .- vec(xref)) / norm(vec(xref))

"""
    check_nrmse(est, ref, tol; label)

Magnitude-align `est` to `ref` (by `‖|ref|‖ / ‖|est|‖`, absorbing the arbitrary global scale
between reconstructions), compute the [`nrmse`](@ref), log it under `label`, and assert it is
below `tol`. Returns the NRMSE.
"""
function check_nrmse(est, ref, tol; label)
    e = nrmse(est .* (norm(abs.(ref)) / norm(abs.(est))), ref)
    @info "$label NRMSE: $e"
    @test e < tol
    return e
end

end

"""
    BenchUtils

Everything the Ristretto harness (`benchmark/run.jl`) and the comparison suite (RistrettoBenchmarking)
share: site configuration, the case catalog with its phantoms, sampling patterns, noise and real
data, Ristretto's reconstruction of each method, the timing function and the result store.

Depends only on Ristretto, GeometricMedicalPhantoms, FFTW, NamedDims, MRITestData, JSON and standard
libraries. It never assumes BART, SigPy or MRIReco are available; those live in the comparison
suite's own bridges.

    include(joinpath(<repo>, "benchmark", "utils", "bench_utils.jl"))
    using .BenchUtils
"""
module BenchUtils

using Dates: Dates, now, @dateformat_str
using FFTW: FFTW, fft, ifft, fftshift, ifftshift
using GeometricMedicalPhantoms: create_shepp_logan_phantom, create_torso_phantom, MRISheppLoganIntensities,
    generate_cardiac_signals, generate_respiratory_signal
using JSON: JSON
using LinearAlgebra: norm
using NamedDims: NamedDimsArray, unname
using Printf: @sprintf
using Random: AbstractRNG, MersenneTwister, randperm
using SHA: sha1
using Serialization: serialize, deserialize
using Statistics: median

using Ristretto
using Ristretto: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo

include("config.jl")
include("phantoms.jl")
include("sampling.jl")
include("noise.jl")
include("cases.jl")
include("real_data.jl")
include("ristretto_methods.jl")
include("harness.jl")
include("results_store.jl")

export load_site_env!, env_flag, ensure_download_path!
export BenchCase, get_case, case_ids, filter_case_ids, SYNTHETIC_CASES, HARNESS_ONLY_CASES, REAL_CASES, small_mode, cine_frames
export ncoils, acceleration, zero_filled, ristretto_acquisition, applicable_methods, METHODS, PDHG_METHODS, RISTRETTO_ONLY_METHODS, penalty_of
export norm_ksp, add_noise, nrmse, mag_nrmse, centred_fft, centred_ifft, multicoil_phantom
export ristretto_reconstructor, ristretto_regularizer, ristretto_algorithm, DEFAULT_LAMBDA, OUTER_ITERATIONS, CG_ITERATIONS, ADMM_RHO
export RADIAL_LAMBDA, RADIAL_ADMM_RHO, default_lambda, admm_rho, PDHG_ITERATIONS, default_maxit
export time_run, timed_runs, git_ref, tree_hash, node_class, recorded_env, ResultsStore

end

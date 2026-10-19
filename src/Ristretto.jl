module Ristretto

using LinearAlgebra
using Random: Random, AbstractRNG
using Base.Cartesian
using ProximalCore
import ProgressMeter
using NamedDims

# Vendored packages, inlined as submodules so Ristretto has no
# unregistered dependencies (registration requires every dependency to be
# registered). ProximalCore is excluded from this: it stays a normal registered
# dependency (see [deps]/[compat] in Project.toml).
include(joinpath(@__DIR__, "..", "deps", "OperatorCore", "src", "OperatorCore.jl"))
include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "src", "AbstractOperators.jl"))
include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "ContourletOperators", "src", "ContourletOperators.jl"))
include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "DSPOperators", "src", "DSPOperators.jl"))
include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "FFTWOperators", "src", "FFTWOperators.jl"))
include(joinpath(@__DIR__, "..", "deps", "NFFT", "src", "NFFT.jl"))
include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "NFFTOperators", "src", "NFFTOperators.jl"))
include(joinpath(@__DIR__, "..", "deps", "AbstractOperators", "WaveletOperators", "src", "WaveletOperators.jl"))
include(joinpath(@__DIR__, "..", "deps", "ProximalOperators", "src", "ProximalOperators.jl"))
include(joinpath(@__DIR__, "..", "deps", "ProximalAlgorithms", "src", "ProximalAlgorithms.jl"))
include(joinpath(@__DIR__, "..", "deps", "StructuredOptimization", "src", "StructuredOptimization.jl"))

using .ProximalOperators
using .ProximalAlgorithms
using .AbstractOperators
using .AbstractOperators: Sum  # resolve ambiguity with ProximalOperators.Sum
using .StructuredOptimization
using .NFFTOperators: NFFTOp

import NestedThreading
using NestedThreading: @budgeted_threads, capacity, with_full_threads, with_restricted_threads, with_thread_budget,
    with_thread_default
using .WaveletOperators: WaveletOp, WT, wavelet
using .ContourletOperators: ContourletOp, NSCTOp, ContourletParams, parabolic_levels
using .FFTWOperators: FFTWOperators, DFT, fftshift_op, ifftshift_op, alternate_sign!
using RecursiveArrayTools: ArrayPartition
using FFTW: FFTW, fft, ifft, fftshift, ifftshift
using FastBroadcast: @..
import Scratch
import PrecompileTools
using ArgCheck: @argcheck
import Adapt
using Printf: @sprintf
using Statistics: quantile, median, mean, std
using Base.Threads: @threads, @spawn, nthreads
using StatsBase: sample, ProbabilityWeights

const ISTA = ProximalAlgorithms.ForwardBackward
const FISTA = ProximalAlgorithms.FastForwardBackward
const ADMM = ProximalAlgorithms.ADMM
const DouglasRachford = ProximalAlgorithms.DouglasRachford
const CG = ProximalAlgorithms.CG
const CGNR = ProximalAlgorithms.CGNR
const POGM = ProximalAlgorithms.POGM
const ChambollePock = ProximalAlgorithms.ChambollePock
const PDHG = ChambollePock
const VuCondat = ProximalAlgorithms.VuCondat
const NCG = ProximalAlgorithms.NonlinearCG
const LBFGS = ProximalAlgorithms.LimitedMemoryBFGS

# Exported names target a non-expert user assembling a reconstruction from the built-in pieces.
# The extension surface (abstract supertypes, interface functions) is `public` but not exported.
# See NAMING.md for the rules governing this split.

# Regularization terms
export L2Image, L1Image, L1Wavelet2D, L1Wavelet3D, L1Contourlet, TotalVariation2D, TotalVariation3D, L1TemporalFourier, LowRank, RankLimit
export AnisotropicTotalVariation2D, AnisotropicTotalVariation3D
export TemporalTotalVariation, JointSparsity, LocallyLowRank, ReferencePrior, NonNegative, BoxConstraint
export SecondOrderTotalVariation2D, SecondOrderTotalVariation3D, MultiScaleLowRank, StructuredLowRank
export EdgePreservingRoughness2D, EdgePreservingRoughness3D, TotalGeneralizedVariation2D, TotalGeneralizedVariation3D
export L0Image, L0Wavelet2D, L0Wavelet3D, PlugAndPlay
# Established second names for two of the terms above (NAMING.md rule 2.1)
export Tikhonov, LLR

# Top-level entry points and configuration
export build_model, reconstruct, ReconstructionConfig, SequentialExecutor, MultiThreadingExecutor
export plan_fft_wisdom
export Silent, ProgressBar, Verbose
export IterationTrace
export BartScaling, FixedScaling, KSpaceNormScaling, MaxScaling, MeasurementBasedScaling, NoiseLevelScaling
export NoScaling, QuantileScaling, StdScaling, SystemMatrixBasedScaling
export ISTA, FISTA, POGM, ADMM, DouglasRachford, CG, CGNR, ChambollePock, VuCondat, NCG, LBFGS
# Established second name for `ChambollePock` (NAMING.md rule 2.1)
export PDHG

# Reconstruction methods
export DirectReconstruction, IterativeReconstruction
export AdjointSensitivity, RootSumSquares, NoCoilCombination
export L2Loss, HardConsistency, NoFidelity
export LinearRamp, StepRamp, Homodyne, PhaseConstrained, POCS
export GRAPPA
export SPIRiT, SPIRiTConsistency
export partial_fourier_band

# Acquisition data and signal models
export AcquisitionInfo, PartitionedKSpace
export TemporalBasis, KSpaceToImage

# Image decomposition
export Component, ReconImage, components, total_image, drop_components
export Header, settag!, gettag, tags

# Preprocessing
export density_compensation, PipeMenonDCF, VoronoiDCF, correct_dcf_edges
export prewhiten, estimate_noise_covariance
export compress_coils, SVDCompression, GeometricCompression
export estimate_sensitivities, SelfCalibrating, AdaptiveCombine, ESPIRiT
export normalize_sensitivity_maps
export correct_gradient_delays, estimate_gradient_delays, OpposingSpokes, RING

# Analysis and simulation
export pseudo_replica, estimate_snr, snr_masks
export simulate_acquisition, coil_sensitivities, add_noise
export UniformRandomSampling, VariableDensitySampling, PoissonDiskSampling, RegularLatticeSampling, PartialFourierSampling, GaussianDistribution, PolynomialDistribution
export create_sampling_pattern, to_displayable_mask
export radial_trajectory, stack_of_stars_trajectory, kooshball_trajectory, phyllotaxis_trajectory, spiral_trajectory, floret_trajectory, sparkling_trajectory
export LinearOrdering, GoldenAngle, TinyGoldenAngle, Archimedean, VariableDensity

# Individual names reexported from dependencies because a non-expert has to type them.
# Never reexport a whole dependency (NAMING.md rule 6.4).
export NamedDimsArray, dimnames, unname   # build the input array
export WT, wavelet                        # L1Wavelet2D(λ; wavelet = WT.db4)
export ContourletParams, parabolic_levels # L1Contourlet

# Extension surface: dispatch on these, subtype them, or implement them for a new component.
# Documented and stable, but not exported.
public CartesianAcquisitionInfo, NonCartesianAcquisitionInfo
public header
public Regularization, ReconstructionMethod, IterativeMethod, DirectMethod
public Scaling, CoilCombination, DataFidelity, Verbosity, ReconstructionExecutor
public Subsampling, VariableDensityDistribution, PartialFourierFilter
public RadialOrdering, SpiralVariant
public is_partitioned, parts, nparts, ragged_dim, to_array_partition
public DensityCompensation, CoilCompression, SensitivityEstimation, GradientDelay
public get_operator, materialize, materialize_with_auxiliaries, materialize_all
public get_affected_dims, scale_regularization, bind_dimensions, calculate
public check_applicable
public fftw_wisdom_path
public get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator
public build_encoding_operator, signal_model_operator, NamedDimsOp, DFT, DEFAULT_ALGORITHMS
public DEVICE_DISABLES_TASK_SPLITTING

include("acquisition_data/partitioned_kspace.jl")
include("acquisition_data/header.jl")
include("acquisition_data/acquisition_info.jl")
include("acquisition_data/cartesian_acquisition_info.jl")
include("acquisition_data/noncartesian_acquisition_info.jl")
include("acquisition_data/acquisition_info_copy.jl")
include("acquisition_data/dimension_utils.jl")
include("acquisition_data/adapt.jl")

include("preprocessing/density_compensation.jl")
include("preprocessing/prewhitening.jl")
include("preprocessing/coil_compression.jl")
include("preprocessing/sensitivity_estimation.jl")
include("preprocessing/sensitivity_normalization.jl")
include("preprocessing/gradient_delays.jl")

include("scaling.jl")
include("utils.jl")
include("threading_utils.jl")

include("encoding/named_dims_op.jl")
include("encoding/host_staging.jl")
include("encoding/contourlet_stack_op.jl")
include("encoding/fftw_wisdom.jl")
include("encoding/fourier_operators.jl")
include("encoding/sensitivity_map_operators.jl")
include("encoding/subsampling_operators.jl")
include("encoding/encoding_operators.jl")

include("regularization/regularization.jl")
include("regularization/image_domain_reg.jl")
include("regularization/wavelet_reg.jl")
include("regularization/contourlet_reg.jl")
include("regularization/total_variation_reg.jl")
include("regularization/temporal_fourier_reg.jl")
include("regularization/temporal_total_variation_reg.jl")
include("regularization/second_order_total_variation_reg.jl")
include("regularization/edge_preserving_reg.jl")
include("regularization/total_generalized_variation_reg.jl")
include("regularization/low_rank_reg.jl")
include("regularization/locally_low_rank_reg.jl")
include("regularization/multi_scale_low_rank_reg.jl")
include("regularization/structured_low_rank_reg.jl")
include("regularization/loraks_matrices.jl")
include("regularization/joint_sparsity_reg.jl")
include("regularization/l0_reg.jl")
include("regularization/constraint_reg.jl")
include("regularization/reference_prior_reg.jl")
include("regularization/plug_and_play_reg.jl")

include("reconstruction/recon_image.jl")
include("export/export.jl")
include("reconstruction/components.jl")
include("reconstruction/methods/coil_combination.jl")
include("reconstruction/data_fidelity.jl")
include("reconstruction/methods/reconstruction_method.jl")
include("reconstruction/methods/direct_reconstruction.jl")
include("reconstruction/methods/cartesian_fourier_ops.jl")
include("reconstruction/methods/iterative_reconstruction.jl")
include("reconstruction/methods/partial_fourier/filters.jl")
include("reconstruction/methods/partial_fourier/homodyne.jl")
include("reconstruction/methods/partial_fourier/phase_constrained.jl")
include("reconstruction/methods/partial_fourier/pocs.jl")
include("reconstruction/methods/grappa.jl")
include("reconstruction/methods/spirit.jl")
include("reconstruction/encoding_for_method.jl")
include("reconstruction/task_splitting/plan.jl")
include("reconstruction/task_splitting/slicing.jl")
include("reconstruction/task_splitting/stacking.jl")
include("reconstruction/task_splitting/execution.jl")
include("reconstruction/verbosity.jl")
include("reconstruction/config.jl")
include("reconstruction/hard_consistency.jl")
include("reconstruction/chambolle_pock_preconditioning.jl")
include("reconstruction/build_model.jl")
include("reconstruction/progress_utils.jl")
include("reconstruction/initial_guess.jl")
include("reconstruction/direct_reconstruct_dispatch.jl")
include("reconstruction/iteration_trace.jl")
include("reconstruction/solve_core.jl")
include("reconstruction/reconstruct.jl")

include("analysis/pseudo_replica.jl")
include("analysis/snr.jl")

include("simulation/subsampling.jl")
include("simulation/sensitivities.jl")
include("simulation/simulate_acquisition.jl")
include("simulation/add_noise.jl")
include("simulation/trajectories.jl")

function __init__()
    _register_nfft_threads!()
    _init_serial_blas_threshold!()
    atexit(_save_fftw_wisdom)
    return nothing
end

include("precompile.jl")

end # module Ristretto

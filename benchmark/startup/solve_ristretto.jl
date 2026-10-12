# Ristretto's row of the startup benchmark: the solve `ristretto_reconstructor` builds in the
# comparison harness (benchmark/utils/ristretto_methods.jl), on the acquisition
# `ristretto_acquisition` builds, with BLAS and FFTW at the thread count as `_setup.jl` sets them.
include(joinpath(@__DIR__, "child_common.jl"))
const T_IMPORT = @elapsed @eval using Ristretto
using LinearAlgebra: BLAS
using FFTW: FFTW

t_setup = @elapsed begin
    BLAS.set_num_threads(NTHREADS)
    FFTW.set_num_threads(NTHREADS)
    case = load_case()
    lines = case.mask[1, :]
    acq = Ristretto.CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(case.kspace[:, lines, :]);
        is3D = false, image_size = (NX, NY), subsampling = (:, lines), shifted_image_dims = (:x, :y),
        sensitivity_maps = NamedDimsArray{(:x, :y, :coil)}(case.smaps),
    )
    method = if METHOD === :wavelet
        IterativeReconstruction(;
            regularization = L1Wavelet2D(LAMBDA; wavelet = Ristretto.WT.db2, levels = WAVELET_LEVELS),
            algorithm = Ristretto.FISTA(; maxit = MAXIT, tol = 0.0), maxit = MAXIT, reltol = 0.0,
        )
    elseif METHOD === :cgsense
        IterativeReconstruction(; regularization = (), algorithm = Ristretto.CGNR(; maxit = MAXIT, tol = 0.0), maxit = MAXIT, reltol = 0.0)
    else
        error("no $METHOD")
    end
end
measure(() -> parent(reconstruct(acq, method; verbosity = Silent())), T_IMPORT, t_setup, case.reference)
